import "mass"
import "thetaPOT"
import "treeProcessIrr"
import "kdTreeIrregularRankK"

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

----------------------------------------------
--- Selecting the best NN from large patch ---
----------------------------------------------

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


def RANN [m] [n] [d] (Tval: i32) (k: i64) (h: i32) (test_set: [m][d]f32) (queries: [n][d]f32) =
  -- Step 1: shift points
  let (t_shifted_points, q_shifted_points) = shiftTandQPoints test_set queries

  -- Setup for loop
  let init_knns = replicate n (replicate k (-1i32, f32.inf))
  -- let height =  ( log2Int (m / 256))

  -- Step 2-6 The loop:
  let new_knns =
    loop curr_nns = init_knns for t < Tval do
      -- Step 2 Perform the pseudo random orthogonal transformation on the test set and quiery set
      let M1 = i64.i32 <| log2Int d
      let transformed_test_set = pseudoRandomOrthogonalTransformation M1 t t_shifted_points
      let transformed_queries  = (pseudoRandomOrthogonalTransformation M1 t q_shifted_points)

      -- Step 3 Build the kd-tree
      let (leaves, indir, median_dims, median_vals, shp_arr) = buildKdTree h transformed_test_set

      -- Step 4 & 5 Search the tree and find new candidates
      in searchForKnns transformed_queries curr_nns
                       leaves indir median_dims median_vals shp_arr
                       h

  -- Step 7 omitted here
  let (k_inds, _) =  unzip <| map (\i_knn -> unzip i_knn) new_knns
  in k_inds


def superRANN [m] [n] [d] (Tval: i32) (k: i64) (h: i32) (test_set: [m][d]f32) (queries: [n][d]f32) =
  -- Step 1: shift points
  --let shifted_points = shiftPoints test_set
  let (t_shifted_points, q_shifted_points) = shiftTandQPoints test_set queries

  -- Setup for loop
  let init_knns_q = replicate n (replicate k (-1i32, f32.inf))
  let init_knns_t = replicate m (replicate k (-1i32, f32.inf))
  -- let height =  ( log2Int (m / 256))

  -- Step 2-6 The loop:
  let (new_knns_q, new_knns_t) =
    loop (curr_knns_q, curr_knns_t)= (init_knns_q, init_knns_t) for t < Tval do
      -- Step 2 Perform the pseudo random orthogonal transformation on the test set and quiery set
      let M1 = i64.i32 <| log2Int d
      let transformed_test_set = pseudoRandomOrthogonalTransformation M1 t t_shifted_points
      let transformed_queries  = (pseudoRandomOrthogonalTransformation M1 t q_shifted_points)

      -- Step 3 Build the kd-tree
      let (leaves, indir, median_dims, median_vals, shp_arr) = buildKdTree h transformed_test_set

      -- Step 4 & 5 Search the tree and find new candidates
      let curr_knns_q' = searchForKnns transformed_queries curr_knns_q
                          leaves indir median_dims median_vals shp_arr
                          h
      let curr_knns_t' = searchForKnns transformed_test_set curr_knns_t
                          leaves indir median_dims median_vals shp_arr
                          h
      in (curr_knns_q', curr_knns_t')

  -- Step 7 perform depth one search "supercharging" on the found knns of queries
  let (knn_inds_q, _) =  unzip <| map (\i_knn -> unzip i_knn) new_knns_q
  let (knn_inds_t, _) =  unzip <| map (\i_knn -> unzip i_knn) new_knns_t
  let supercharging =
    let (super_knn_inds_seq) =
      loop (curr_knns_q) = (new_knns_q) for i < k do
        let k_inds = map (\knn_ind_q -> map (\q -> knn_inds_t[knn_ind_q[i],q]) (iota k)) knn_inds_q
        let k_points = map (\inds -> map (\ind ->
                                        let test_point = map (\k_ind -> test_set[ind, k_ind]) (iota d)
                                        in (ind, test_point)
                                        ) inds) k_inds
        in  map3 (\refs query query_knn -> bruteForce query query_knn refs ) k_points queries curr_knns_q
    in super_knn_inds_seq

  let (super_knn_inds, _) =  unzip <| map (\i_knn -> unzip i_knn) supercharging
  in (super_knn_inds)


entry main [m] [n] [d] (Tval: i32) (k: i64) (h: i32) (test_set: [m][d]f32) (queries: [n][d]f32) =
  RANN Tval k h test_set queries

entry mainSuper [m] [n] [d] (Tval: i32) (k: i64) (h: i32) (test_set: [m][d]f32) (queries: [n][d]f32) =
  superRANN Tval k h test_set queries
