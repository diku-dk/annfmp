let sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    loop (res) = (0.0f32) for (x,y) in (zip xs ys) do
        let z = x-y in res + z*z

-- OLD VERSION
let bruteForce [m][d][k] (query: [d]f32)
                         (knns0: [k](i32,f32))
                         (beg: i32, refs : [m][d]f32)
                       : [k](i32,f32) =
    loop (knns) = (copy knns0)
      for i < m do
        let dist = sumSqrsSeq query (refs[i]) in
        if dist > knns[k-1].1 then knns -- early exit
        else let ref_ind = (i32.i64 i)+beg in
             let (_, _, knns') =
               loop (dist, ref_ind, knns) for j < k do
                 let cur_nn = knns[j].1  in
                 if dist >= cur_nn
                 then (dist, ref_ind, knns)
                 else let tmp_ind = knns[j].0
                      let knns[j] = (ref_ind, dist)
                      let ref_ind = tmp_ind
                      in  (cur_nn, ref_ind, knns)
             in  knns'

let sortPartSortedSeqs [k] (knn: [k](i32,f32)) : [k](i32,f32) =
  -- now knn contains the neighbors in two partially ordered sequences:
  -- one starting at beginning and one starting at the end
  -- we need to sort them
  -- let knn = intrinsics.opaque (copy knn0)
  let (res, _, _) =
    loop (knn_sort, beg, end) = (replicate k (-1i32, f32.highest), 0, k-1)
      for i < k do
        let (next_el, beg', end') =
          if knn[beg].1 < knn[end].1
          then (knn[beg], beg+1, end)
          else (knn[end], beg, end-1)
        let knn_sort[i] = next_el
        in  (knn_sort, beg', end')
  in  res

-- OLD VERSION
-- query: query point to be searched for
-- knn0:  initial knns
-- beg:   index pointing at beginning of leaf
-- refs:  2D list of point in leaf
let bruteForcePar [m][d][k] (query: [d]f32)
                         (knn0: [k](i32,f32))
                         (beg: i32, refs : [m][d]f32)
                       : [k](i32,f32) = #[unsafe]
  let knn = copy knn0
  let dists = map (sumSqrsSeq query) refs -- euclidian distances
  let cycle = true
  let j = 0i32
  let (_, knn, _, _) =
    loop (dists, knn, j, cycle)
      while cycle && (j < (i32.i64 k)) do
        let (min_ind, min_val) =
          reduce_comm (\ (i1,v1) (i2,v2) ->
                        if v1 < v2 then (i1, v1) else
                        if v1 > v2 then (i2, v2) else
                        (if i1 <= i2 then i1 else i2, v1)
                      ) (m, f32.inf) (zip (iota m) dists)

        in  if min_val < (knn[k-1-(i64.i32 j)].1)
            then  let dists[min_ind] = f32.highest
                  let knn[k-1-(i64.i32 j)] = (beg+(i32.i64 min_ind), min_val)
                  in  (dists, knn, j+1, true)
            else  (dists, knn, j, false)
  let knn_sort = sortPartSortedSeqs knn
  in  knn_sort

let bruteForceParIrreg [m][d][k]
      (query: [d]f32)
      (knn0: [k](i32,f32))
      (beg: i64)
      (len: i64)
      (avg_leafsize: i64)
      (refs: [m][d]f32)
    : [k](i32,f32) = #[unsafe]
  let Q = 4i64
  let B = (avg_leafsize + Q - 1) / Q
  in
  loop knn = copy knn0
    for qq < (len + avg_leafsize - 1) / avg_leafsize do
      let offset = qq * avg_leafsize
      let fdist q =
            let ind = offset + q + beg
            in  if ind < len + beg
                then sumSqrsSeq query refs[ind]
                else f32.highest
      let dists = map fdist (0 ... B-1) ++
                  map fdist (B ... 2*B-1) ++
                  map fdist (2*B ... 3*B-1) ++
                  map fdist (3*B ... 4*B-1)
      let cycle = true
      let j = 0i32
      let (_, knn, _, _) =
        loop (dists, knn, j, cycle)
          while cycle && (j < (i32.i64 k)) do
            let fmap li =
                loop (midx, mval) = (i32.highest, f32.highest)
                  for i < Q do
                    let pt_idx = i * B + li
                    let dis = dists[pt_idx]
                    in if dis < mval then (i32.i64 (offset + pt_idx), dis) else (midx, mval)
            let (min_ind, min_val) =
              map fmap (iota B) |>
              reduce_comm (\ (i1,v1) (i2,v2) ->
                            if v1 < v2 then (i1, v1) else
                            if v1 > v2 then (i2, v2) else
                            (if i1 <= i2 then i1 else i2, v1)
                          ) (i32.highest, f32.highest)
            in  if min_val < (knn[k-1-(i64.i32 j)].1)
                then  let dists[min_ind-(i32.i64 offset)] = f32.highest
                      let knn[k-1-(i64.i32 j)] = (min_ind + i32.i64 beg, min_val)
                      in  (dists, knn, j+1, true)
                else  (dists, knn, j, false)
      let knn_sort = sortPartSortedSeqs knn
      in  knn_sort

-- SEGMENTED VERSION
-- query: query point to be searched for
-- knn0:  Initial knns
-- beg:   Index pointing at beginning of leaf
-- len:   Length of the leaf
-- refs:  2D list of points

-- Result:  Sorted Knn

let bruteForceSegPar [m][d][k] (query: [d]f32)
                         (knn0: [k](i32,f32))
                         (beg: i64)
                         (len: i64)
                         (avg_leafsize: i64)
                         (ref_pts: [m][d]f32)
                       : [k](i32,f32) = #[unsafe]
  let B = avg_leafsize -- avg_leafsize
  let knn = copy knn0
  let visited = replicate k (-1i64)
  let cycle = true
  let j = 0i32
  let (_, knn, _, _) =
    loop (visited, knn, j, cycle)
      while cycle && (j < (i32.i64 k)) do
        let (min_ind, min_val) =
          map (\li ->
          loop (midx, mval) = (len, f32.inf)
          for i < (len + B - 1)/B do
            let pt_idx = i * B + li
            let seen = loop found = false for vi < k do
                           found || (visited[vi] == pt_idx)
            in if pt_idx < len && !seen
              then let dis = sumSqrsSeq query ref_pts[beg + pt_idx]
                in if dis < mval then (pt_idx, dis) else (midx, mval)
              else (midx, mval)
          ) (iota B) |> reduce_comm (\ (i1,v1) (i2,v2) -> if v1 <= v2 then (i1,v1) else (i2,v2)) (len, f32.inf)
--          reduce_comm (\ (i1,v1) (i2,v2) ->
--                        if v1 < v2 then (i1, v1) else
--                        if v1 > v2 then (i2, v2) else
--                        (if i1 <= i2 then i1 else i2, v1)
--                      ) (len, f32.inf) (zip (iota len) dists)
        in  if min_val < (knn[k-1-(i64.i32 j)].1)
            then  let visited[i64.i32 j] = min_ind
                  let knn[k-1-(i64.i32 j)] = (i32.i64 beg + i32.i64 min_ind, min_val)
                  in  (visited, knn, j+1, true)
            else  (visited, knn, j, false)
  let knn_sort = sortPartSortedSeqs knn
  in  knn_sort

-------------------------------------------
---- ENTRY POINTS and utilities
-------------------------------------------

def imap2intra f as bs =
  #[incremental_flattening(only_intra)] map2 f as bs

def imap3intra as bs cs f =
  #[incremental_flattening(only_intra)] map3 f as bs cs

def imap4intra f as bs cs ds =
  #[incremental_flattening(only_intra)] map4 f as bs cs ds


def kk : i64 = 8i64
def d  : i64 = 16i64

entry mk_input
        (leaf_size:   i64)
        (num_leaves:  i64)
        (num_queries: i64)
      : (i64, i64, [num_queries][d]f32, [num_queries]i32, [num_queries][kk]i32, [num_queries][kk]f32, [num_leaves*leaf_size][d]f32)
      =
  let begs = map (\i -> i32.i64 ((i % num_leaves)*leaf_size)) (iota num_queries)
  let query = [ 0.1f32, 0.2f32, 0.3f32, 0.4f32, 0.5f32, 0.6f32, 0.7f32, 0.8f32
              , 0.8f32, 0.7f32, 0.6f32, 0.5f32, 0.4f32, 0.3f32, 0.2f32, 0.1f32
              ] :> [d]f32
  let queries = replicate num_queries query
  let knn_ind = map (\i -> i32.i64 (0 - i - 1)) (iota kk)
  let knn_val = replicate kk f32.highest
  let knn_inds= replicate num_queries knn_ind
  let knn_vals= replicate num_queries knn_val
  let refs = tabulate_2d num_leaves leaf_size
        (\ _ii i -> let delta = f32.i64 (i+1) / f32.i64 leaf_size
                    in  map (*delta) query
        ) |> flatten
  in  (leaf_size, num_leaves, queries, begs, knn_inds, knn_vals, refs)


-- Brute Force Intra-Parallel Regular (kk = 8, d = 16, n = 1024*1024=1048576, m = leaf_size * num_leaves = 1024 * 1024 = 1048576)
-- ==
-- entry: runBruteForceReg runBruteForceIrreg runBruteForceIrregCos
--
-- "fix-pattern-512-512" script input { mk_input 512i64 512i64 262144i64 }
-- "fix-pattern-1024-1024" script input { mk_input 1024i64 1024i64 1048576i64 }

entry runBruteForceReg [n][m][d]
        (leaf_size:  i64)
        (num_leaves: i64)
        (queries:  [n][d]f32)
        (begs:     [n]i32)
        (knn_inds: [n][kk]i32)
        (knn_vals: [n][kk]f32)
        (refs':     [m][d]f32)
      : ([n][kk]i32, [n][kk]f32) =
  let refs = unflatten (refs' :> [num_leaves*leaf_size][d]f32)
  let knns0 = map2 zip knn_inds knn_vals
--  let begs = map (\i -> i32.i64 ((i % num_leaves)*leaf_size)) (iota n)
  let f query knn0 beg =
    let ind = beg / i32.i64 leaf_size
    let ref = refs[ind]
    in  bruteForcePar query knn0 (beg, ref)
  let knns =  imap3intra queries knns0 begs f
  in  map unzip knns |> unzip

entry runBruteForceIrreg [n][m][d]
        (leaf_size:  i64)
        (num_leaves: i64)
        (queries:  [n][d]f32)
        (begs:     [n]i32)
        (knn_inds: [n][kk]i32)
        (knn_vals: [n][kk]f32)
        (refs':     [m][d]f32)
      : ([n][kk]i32, [n][kk]f32) =
  let ref_pts = refs' :> [num_leaves*leaf_size][d]f32
  let knns0 = map2 zip knn_inds knn_vals
  --
  let f query knn0 beg =
    bruteForceSegPar query knn0 beg leaf_size 256 ref_pts
  --
  let knns =  imap3intra queries knns0 (map i64.i32 begs) f
  in  map unzip knns |> unzip

entry runBruteForceIrregCos [n][m][d]
        (leaf_size:  i64)
        (num_leaves: i64)
        (queries:  [n][d]f32)
        (begs:     [n]i32)
        (knn_inds: [n][kk]i32)
        (knn_vals: [n][kk]f32)
        (refs':     [m][d]f32)
      : ([n][kk]i32, [n][kk]f32) =
  let ref_pts = refs' :> [num_leaves*leaf_size][d]f32
  let knns0 = map2 zip knn_inds knn_vals
  --
  let f query knn0 beg =
    bruteForceParIrreg query knn0 beg leaf_size leaf_size ref_pts
  --
  let knns =  imap3intra queries knns0 (map i64.i32 begs) f
  in  map unzip knns |> unzip


-- Validating Brute Force Intra-Parallel Regular vs Irregular
-- ==
-- entry: equivRegIrreg
--
-- "fix-pattern-512-512" script input { mk_input 512i64 512i64 262144i64 }
-- output { true }
-- "fix-pattern-1024-1024" script input { mk_input 1024i64 1024i64 1048576i64 }
-- output { true }


entry equivRegIrreg [n][m][d]
        (leaf_size:  i64)
        (num_leaves: i64)
        (queries:  [n][d]f32)
        (begs:     [n]i32)
        (knn_inds: [n][kk]i32)
        (knn_vals: [n][kk]f32)
        (refs':     [m][d]f32)
      : bool =
  let refs = unflatten (refs' :> [num_leaves*leaf_size][d]f32)
  let knns0 = map2 zip knn_inds knn_vals
  let freg query knn0 beg =
    let ind = beg / i32.i64 leaf_size
    let ref = refs[ind]
    in  bruteForcePar query knn0 (beg, ref)
  let knns_reg =  imap3intra queries knns0 begs freg |> opaque
  --
  let ref_pts = refs' :> [num_leaves*leaf_size][d]f32
  let f1 query knn0 beg =
    bruteForceSegPar query knn0 beg leaf_size 256 ref_pts
  let knns_irreg1 =  imap3intra queries knns0 (map i64.i32 begs) f1 |> opaque
  --
  let f2 query knn0 beg =
    bruteForceParIrreg query knn0 beg leaf_size leaf_size ref_pts
  let knns_irreg2 =  imap3intra queries knns0 (map i64.i32 begs) f2 |> opaque

  let ok1 = map2 (==) knns_reg knns_irreg1 |> reduce (&&) true
  let ok2 = map2 (==) knns_reg knns_irreg2 |> reduce (&&) true
  in  (ok1 && ok2)
