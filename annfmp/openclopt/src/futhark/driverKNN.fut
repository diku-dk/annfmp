import "lib/github.com/diku-dk/sorts/radix_sort"

import "buildKDtree"
import "brute-force"
import "kd-traverse"

let sortQueriesByLeavesRadix [n] (num_bits: i32) (leaves: [n]i32) : ([n]i32, [n]i32) =
  unzip <| radix_sort_by_key (\(l,_) -> l) num_bits i32.get_bit (zip leaves (iota n))

----------------------------
--- The fixed value of K ---
----------------------------
let kk = 8i32

---------------
--- Helpers ---
---------------
let getHeightPpl (q: i32) (m: i32) : (i32, i32, i32) =
  let num_nodes  = q
  let num_leaves = num_nodes + 1
  let h = (log2 num_leaves) - 1
  let ppl = m / num_leaves
  in  (h, ppl, num_leaves)

---------------------------------------
--- Breaking the image into patches ---
---------------------------------------
entry mkImgPatches [h][w][c] (p: i32) (img: [h][w][c]i32) : [][]u8 =
  let n_cols = w - p + 1
  let n_rows = h - p + 1
  let ppc = p*p*c
  let n_all  = n_cols*n_rows

  let mkPatch ii jj = 
  	tabulate_2d p p
  		(\i j ->
  			map (\k -> u8.i32 (img[ii+i, jj+j, k])) (iota c)
  		)
  let res5d = tabulate_2d n_rows n_cols mkPatch
  let res2d = map (\(patch: [p][p][c]u8) -> (flatten (flatten patch)) :> [ppc]u8)
  				        ( (flatten res5d) :> [n_all][p][p][c]u8)
  in  res2d

----------------------------------------------
--- Reducing the dimensionality of patches ---
----------------------------------------------
entry reducePatchDim [n][d][d_red] (img: [n][d]u8) (comps: [d_red][d]f32) (means: [d]f32) : [n][d_red]f32 =
  map (\ (patch: [d]u8) ->
        map(\ (comp: [d]f32) ->
              f32.sum <| map3 (\p c m -> ( (f32.u8 p) - m ) * c) patch comp means
           ) comps
      ) img

-- this reduces the dimensionality of patches without
-- manifesting the array of large patches but is slower!
let reducePatchDim_OLD [d][d_red][h][w][c] (img: [h][w][c]i32) (comps: [d_red][d]f32) (means: [d]f32) : [][d_red]f32 =
  let pp = d / c
  let p  = i32.f32 (f32.sqrt (f32.i32 pp))
  let n_cols = w - p + 1
  let n_rows = h - p + 1
  let n_all  = n_cols*n_rows
  in
  map (\ ind_patch ->
        map (\ (comp: [d]f32) ->
              let ii = intrinsics.opaque (ind_patch / n_cols)
              let jj = intrinsics.opaque (ind_patch - (ii * n_cols)) 
              in  f32.sum <|
                  map3(\ (ijk: i32) (cmp: f32) (m: f32) ->
                        let ij = ijk / c
                        let k  = ijk - (ij * c)
                        let i  = ij / p
                        let j  = ij - (i * p)
                        let v_img = f32.i32 (img[ii+i, jj+j, k])
                        in  ( v_img - m ) * cmp
                      ) (iota d) comp means
            ) comps
      ) (iota n_all)


----------------------------------------------
--- Selecting the best NN from large patch ---
----------------------------------------------

entry selectBestNN_BAD [n][w1][w2][h1][h2][c] 
                    (p: i32) (knn_inds: [n][kk]i32)
                    (imgA: [h1][w1][c]f32) (imgB: [h2][w2][c]f32) 
                  : ([n]i32, [n]f32) =
  (knn_inds[:,0], replicate n 0.0f32)

entry selectBestNN [n][w1][w2][h1][h2][c]
                    (p: i32) (knn_inds: [n][kk]i32)
                    (imgA: [h1][w1][c]i32) (imgB: [h2][w2][c]i32)
                  : ([n]i32, [n]f32, f32) =
  let n_colsA = w1 - p + 1
  --let n_rowsA = h1 - p + 1
  let n_colsB = w2 - p + 1
  --let n_rowsB = h2 - p + 1
  let patch_len = p*p*c
  let (nn_inds, nn_dsts) = unzip <|
    map2(\knns indA -> 
          let y = indA / n_colsA
          let x = indA - y * n_colsA
          let query = map (\ ijk ->
                            let ij = ijk / c
                            let k  = ijk - ij*c
                            let i = ij / p
                            let j = ij - i*p
                            in  f32.i32 (imgA[y+i, x+j, k])
                          ) (iota patch_len)
          let (nn_ind, nn_dst) = (-1i32, f32.inf) in
          loop (nn_ind, nn_dst) for q < kk do
            let indB = knns[q]
            let ii = indB / n_colsB
            let jj = indB - ii * n_colsB
            let dst = f32.sum <|
              map (\ ijk ->
                    let ij = ijk / c
                    let k  = ijk - ij*c
                    let i = ij / p
                    let j = ij - i*p
                    let b_v = f32.i32 (imgB[ii+i, jj+j, k])
                    let a_v = query[ijk]
                    let d = b_v - a_v
                    in  d*d
                  ) (iota patch_len)
            in  if dst < nn_dst
                then (indB, dst)
                else (nn_ind, nn_dst)
        ) knn_inds (iota n)
  let err = reduce (+) 0.0f32 nn_dsts
  in  (nn_inds, nn_dsts, f32.sqrt err)

-----------------------------
--- Building the k-d Tree ---
-----------------------------

-- ==
-- entry: buildKDtree
--
-- compiled random input { 256i32 [2097152][7]f32 }

entry buildKDtree [m][d] (defppl: i32) (input: [m][d]f32) =
    let (height, num_inner_nodes, ppl, m') = computeTreeShape m defppl
    let (leafs, indir, median_dims, median_vals, clanc_eqdim) =
          mkKDtree height num_inner_nodes m' input
    let orig2leaf = scatter (replicate m' (-1i32)) indir
                            (map (\i -> i / ppl) (iota m'))
    in  (height, num_inner_nodes, m', leafs, indir, orig2leaf, median_dims, median_vals, clanc_eqdim)

--------------------------------------------------------------
--- Finding the natural leaf to which the query belongs to ---
--------------------------------------------------------------
let findNaturalLeaves [m][d][q][n] (k: i32)
                                (ref_pts:  [m][d]f32) 
                                (median_dims: [q]i32)
                                (median_vals: [q]f32)
                                (queries:  [n][d]f32) :
                                (*[n][k]i32, *[n][k]f32, *[n]i32) =
  let (h, ppl, num_leaves) = getHeightPpl q m
  let leaves = unflatten num_leaves ppl ref_pts

  let query_leaves0 = map (findLeaf median_dims median_vals h) queries
  let (query_leaves, query_inds) = sortQueriesByLeavesRadix (h+1) query_leaves0
  let queries = gather2D queries query_inds
  let knns0 = map2(\leaf_ind query -> bruteForce query (replicate k (-1i32, f32.highest))
                                                  (leaf_ind*ppl, leaves[leaf_ind])
                  ) query_leaves queries
  let dummy = replicate n (replicate k (-1i32, f32.highest))
  let knns0' = scatter2D dummy query_inds knns0
  let (knn_inds, knn_vals) = unzip <| map unzip knns0'
  in  (knn_inds, knn_vals, query_leaves0)

entry findNaturalLeavesFixK [m][d][q][n]
                          (ref_pts:  [m][d]f32) 
                          (median_dims: [q]i32)
                          (median_vals: [q]f32)
                          (queries: [n][d]f32) :
                          (*[n][kk]i32, *[n][kk]f32, *[n]i32) =
  findNaturalLeaves kk ref_pts median_dims median_vals queries

-----------------------------
--- Finding the exact knn ---
-----------------------------
let exactKnnOld [m][q][d][n][k]
              (ref_pts: [m][d]f32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (nat_leaves :[n]i32)
              (knns:    [n][k](i32,f32)) : ([n][k](i32,f32), i32) =
  let (h, ppl, num_leaves) = getHeightPpl q m
  let leaves = unflatten num_leaves ppl ref_pts

  let last_leaves = nat_leaves
  let stacks = replicate n 0i32
  let dists  = replicate n 0.0f32
  let ord_knns = copy knns
  let query_inds = iota n
  let n' = n
  let i = 0i32

  let (_,_,_,_,_, ord_knns', _, loop_count, _) =
      loop (queries, knns, last_leaves, stacks, dists, ord_knns, query_inds, i, n')
        while n' > 0 do
        --while (length queries > 0) do
          let wnns = map (\arr -> arr[k-1].1) knns
          let (new_leaves, new_stacks, new_dists) = unzip3 <|
            map2 (traverseOnce h kd_tree) (zip queries wnns) (zip3 last_leaves stacks dists)

          let (inds_part, n'') = partition2Ind <| map (\i -> new_leaves[i] < num_leaves) (iota n')
          let qinds_part = gather1D query_inds inds_part
          
          let (inds1, inds2) = split n'' <| zip inds_part qinds_part
          let (iota_valid, query_inds') = unzip inds1
          let (iota_done,  qinds_updt ) = unzip inds2

          -- get valid part of arrays
          let queries' =  gather2D queries iota_valid
          let knns'    =  gather2D knns    iota_valid
          let (new_leaves', new_stacks', new_dists') = unzip3 <|
                          gather1D (zip3 new_leaves new_stacks new_dists) iota_valid

          -- update global knns
          let ord_knns' = scatter2D ord_knns qinds_updt (gather2D knns iota_done)

          -- do brute force
          let knns'' = map3 (\query knn leaf_ind -> bruteForcePar query knn (leaf_ind*ppl, leaves[leaf_ind]))
                            queries' knns' new_leaves'
          in  (queries', knns'', new_leaves', new_stacks', new_dists', ord_knns', query_inds', i+1, n'')
  in  (ord_knns', loop_count)

let exactKnn [m][q][d][n][k]
              (ref_pts: [m][d]f32)
              (kd_tree: [q](i32,f32,i32))
              (queries: [n][d]f32)
              (nat_leaves :[n]i32)
              (knns:   *[n][k](i32,f32)) : (*[n][k](i32,f32), i32) =
  let (h, ppl, num_leaves) = getHeightPpl q m
  let leaves = unflatten num_leaves ppl ref_pts

  let last_leaves = nat_leaves
  let stacks = replicate n 0i32
  let dists  = replicate n 0.0f32
  let n' = n
  let i = 0i32

  let (ord_knns', _, _, _, loop_count, _) =
      loop  (knns, last_leaves, stacks, dists, i, n')
        while n' > 0 do
          let wnns = map (\arr -> arr[k-1].1) knns
          let (new_leaves, new_stacks, new_dists) = unzip3 <|
            map2 (traverseOnce h kd_tree) (zip queries wnns) (zip3 last_leaves stacks dists)

          let n'' = map (\leaf_ind -> if leaf_ind < num_leaves then 1i32 else 0i32) new_leaves
                 |> reduce_comm (+) 0i32
                 |> intrinsics.opaque

          -- do brute force
          let knns' = map3 (\query knn leaf_ind ->
                              -- let knn = intrinsics.opaque <| copy knn0
                              let count = if leaf_ind < num_leaves then 1i32 else 0i32
                              in  loop (knn) for _j < count do
                                      bruteForcePar query knn (leaf_ind*ppl, leaves[leaf_ind])
                           ) queries knns new_leaves
                    |> intrinsics.opaque

          in  (knns', new_leaves, new_stacks, new_dists, i+1, n'')
  in  (ord_knns', loop_count)

entry exactKnnFixK [m][q][d][n]
              (ref_pts: [m][d]f32)
              (median_dims: [q]i32)
              (median_vals: [q]f32)
              (prev_eqdims: [q]i32)
              (s: i32)
              (queries: [n][d]f32)
              (nat_leaves :[n]i32)
              (knn_is: *[n][kk]i32)
              (knn_vs: *[n][kk]f32) : (*[n][kk]i32, *[n][kk]f32, i32) =
  let knns = map2 zip knn_is[:s] knn_vs[:s]
  let kd_tree = zip3 median_dims median_vals prev_eqdims
  let (ord_knns, loop_count) =
    exactKnn ref_pts kd_tree (queries[:s]) (nat_leaves[:s]) knns

  let (knn_inds, knn_dsts) = unzip <| map unzip <| ord_knns
  let knn_is[:s] = knn_inds
  let knn_vs[:s] = knn_dsts
  
  in (knn_is, knn_vs, loop_count)

------------------------------
--- Propagatingation Phase ---
------------------------------
let estimateIndex [m] (n_rows: i32) (n_cols: i32)
                  (indir: [m]i32) (orig2leaf: [m]i32)
                  (par_ind: i32) : i32 =
  let orig_par_ind = indir[par_ind]
  let orig_par_y = orig_par_ind / n_cols
  let orig_par_x = orig_par_ind - orig_par_y * n_cols
  let cand_x = orig_par_x
  let cand_y = if orig_par_y < (n_rows-1) then orig_par_y+1 else n_rows-1
  let orig_cand_ind = cand_y * n_cols + cand_x
  let cand_leaf_ind = orig2leaf[orig_cand_ind]
  in  cand_leaf_ind

let propagate [m][d][nr][nc][k]
              (h: i32)
              (ref_pts:   [m][d]f32)
              (indir:     [m]i32)
              (orig2leaf: [m]i32)
              (queries: [nr][nc][d]f32)
              (nat_leaves :[nr][nc]i32)
              (knns: *[nr][nc][k](i32,f32))
            : *[nr][nc][k](i32,f32) =
  let num_leaves = 1 << (h+1)
  let ppl = m / num_leaves
  let leaves = unflatten num_leaves ppl ref_pts

  let knns' =
    loop (knns) for im1 < nr-1 do
      let i = im1+1
      let upw_knn_inds = map (map (.0)) (knns[im1])
      let cur_nodes = intrinsics.opaque <| copy <| knns[i]

      -- gather leaves from the neighbor directly above:
      let (n_leavess, to_search_leavess) = intrinsics.opaque <| unzip <|
        map2(\ (par_inds0: [k]i32) (nat_leaf: i32) : (i32, [k]i32) ->
                  let par_inds = intrinsics.opaque (copy par_inds0)
                  let u_leafs = replicate k (-1i32)
                  let n_inds  = 0i32
                  let (n_inds, u_leafs) =
                    loop (n_inds, u_leafs)
                      for q < k do
                        -- let par_ind = par_inds[q] / ppl
                        let par_ind = 
                          estimateIndex nr nc indir orig2leaf (par_inds[q])

                        let not_found = par_ind != nat_leaf
                        let j = 0i32
                        let (_,not_found) =
                          loop (j,not_found)
                            while not_found && j < n_inds do
                              if u_leafs[j] == par_ind
                              then (j,   false)
                              else (j+1, true)
                        in  if not_found
                            then  let u_leafs[n_inds] = par_ind
                                  in  (n_inds+1, u_leafs)
                            else  (n_inds, u_leafs)
                  in  (n_inds, u_leafs)
            ) upw_knn_inds (nat_leaves[i])

      let new_knn_row = intrinsics.opaque <|
        map4(\ (n_inds: i32) (u_leafs: [k]i32) (knn0: [k](i32,f32)) (query: [d]f32) ->
                let knn = intrinsics.opaque <| copy knn0
                in  loop (knn) for q < n_inds do
                      let leaf_ind = u_leafs[intrinsics.opaque(q)]
                      in  bruteForcePar query knn (leaf_ind*ppl, leaves[leaf_ind])
            ) n_leavess to_search_leavess cur_nodes (queries[i])

      let knns[i] = new_knn_row
      in  knns
  in  knns'
      
--      let new_knn_row = 
--          map4(\(par_inds0: [k]i32) (knn0: [k](i32,f32)) (nat_leaf: i32) (query: [d]f32) ->
--                  let par_inds = intrinsics.opaque (copy par_inds0)
--                  let u_leafs = replicate k (-1i32)
--                  let n_inds  = 0i32
--                  let (n_inds, u_leafs) =
--                    loop (n_inds, u_leafs)
--                      for q < k do
--                        let par_ind = par_inds[q] / ppl
--                        let not_found = par_ind != nat_leaf
--                        let j = 0i32
--                        let (_,not_found) =
--                          loop (j,not_found)
--                            while not_found && j < n_inds do
--                              if u_leafs[j] == par_ind
--                              then (j,   false)
--                              else (j+1, true)
--                        in  if not_found
--                            then  let u_leafs[n_inds] = par_ind
--                                  in  (n_inds+1, u_leafs)
--                            else  (n_inds, u_leafs)
--                  let knn = intrinsics.opaque (copy knn0)
--                  in --map2 (\j (i,v) -> (i*u_leafs[j], v*2.0f32)) (iota k) knn
--                  loop (knn) for q < n_inds do
--                    let leaf_ind = u_leafs[q]
--                    in  bruteForce query knn (leaf_ind*ppl, leaves[leaf_ind])
--              ) upw_knn_inds (knns[i]) (nat_leaves[i]) (queries[i])

entry propagateFixK [m][d][n]
              (h: i32)
              (nr: i32) -- assumes `nr` equaly divides `n` 
              (ref_pts: [m][d]f32)
              (indir:   [m]i32)
              (orig2leaf: [m]i32)
              (queries0: [n][d]f32)
              (nat_leaves0 :[n]i32)
              (knn_inds: *[n][kk]i32)
              (knn_dsts: *[n][kk]f32)
            : (*[n][kk]i32, *[n][kk]f32) =
  let nc = n / nr
  let queries = unflatten nr nc queries0
  let nat_leaves = unflatten nr nc nat_leaves0
  let knns = unflatten nr nc <| map2 zip knn_inds knn_dsts

  let knns' = propagate h ref_pts indir orig2leaf queries nat_leaves knns
  let knns_flat' = flatten knns' :> [n][kk](i32,f32)
  let (knn_inds', knn_dsts') = unzip <| map unzip knns_flat' 
            

  let knn_inds'' = map (\kinds -> map (\ind -> indir[ind]) kinds) knn_inds'
  in (knn_inds'', knn_dsts')