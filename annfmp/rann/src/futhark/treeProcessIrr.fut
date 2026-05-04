-- Randomized Approximate K-D Trees
-- ==
-- compiled input @ dataProcess.in

-- output @ LocVolCalib-data/small.out

import "kdTreeIrregularRankK"
import "lib/github.com/diku-dk/sorts/radix_sort"

def sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    loop (res) = (0.0f32) for (x,y) in (zip xs ys) do
        let z = x-y in res + z*z

-- Bruteforce knn:
--    MAYBE: bruteforce should exit if it encounters a point it has seen before
--     This is for when the algorithm is run where the points are rotated with different rnd vals
--     since some of the points that will be compared might be the same.
--     Solution1: save knns on each loop of step 2-5. And search these in step 6
--     Solution2: Do not update when a point that is already in the knn is encounter (when refs[i].0 is in one knn[j].0) 
def bruteForce [m][d][k] (query: [d]f32) 
                         (knns0: [k](i32,f32))
                         (refs: [m](i32,[d]f32))
                       : [k](i32,f32) =
    loop (knns) = (copy knns0)
      for i < i32.i64 m do
        let dist = sumSqrsSeq query (refs[i].1) in
        if dist >= knns[k-1].1 then knns -- early exit
        else let ref_ind = refs[i].0 in
              let (_, _, knns') =
                loop (dist, ref_ind, knns) for j < k do
                  let cur_nn = knns[j] in
                  if cur_nn.0 == ref_ind 
                       then (f32.inf, -1, knns) -- already encountered
                  else if dist >= cur_nn.1
                       then (dist, ref_ind, knns)
                  else let tmp_ind = cur_nn.0
                       let knns[j] = (ref_ind, dist)
                       let ref_ind = tmp_ind
                       in  (cur_nn.1, ref_ind, knns)
              in  knns'

-- Given the median dimensions and values for each internal k-d tree node,
--   finds the leaf to which the query naturally belongs to.
def findLeaf [q][d] (median_dims: [q]i32) (median_vals: [q]f32)
                    (height: i32) (query: [d]f32) =
  let leaf =
    loop (node_index) = (0)
      while !( node_index >= ((1 << (height)) - 1)) do -- Maybe remove ! and replace >= with < 
        if query[median_dims[node_index]] < median_vals[node_index]
            then
                (node_index+1)*2-1
            else
                (node_index+1)*2
  let leaf_val = leaf - i32.i64 q
  in i64.i32 leaf_val

def reverseBit (num: i64) (bit: i64) : i64 =
  let bit_val  = 1 << bit
  let bit_rm = (num / bit_val) % 2
  let rev_val = if bit_rm == 1 then -bit_val
                                else  bit_val
  in num + rev_val
    
def log2Int (n : i64) : i32 =
  let (_, res) =
    loop (n, r) = (n, 0i32)
      while n > 1 do
        (n >> 1, r+1)
  in res 

def searchForKnns [m] [n] [d] [k] 
                  (queries: [n][d]f32) (init_knns: *[n][k](i32, f32)) 
                  (leaves: [m][d]f32) (indir: [m]i32) (median_dims: []i32) 
                  (median_vals: []f32) (shape_arr: []i32) (height: i32) =
  let leaves_shp = 
    let beg = i64.i32 (1 << height) - 1
    let end = beg + (i64.i32 (1 << (height )))
    in shape_arr[beg:end]
  let scInc_leaves_shp = scan (+) 0i32 leaves_shp |> map (i64.i32)
  let scExc_leaves_shp = [0i64] ++ (scInc_leaves_shp[:((length leaves_shp) - 1 )]) :> []i64

  let queries_init_leafs = map (findLeaf median_dims median_vals height) queries

  let (sorted_query_with_ind, sorted_query_leaf) = 
    let q_with_ind = zip queries (iota n) 
    let q_ind_with_leaf = zip q_with_ind queries_init_leafs
    in (radix_sort_int_by_key (\(_,l) -> l) (log2Int (length leaves)) i64.get_bit q_ind_with_leaf) |> unzip
  
  let (sorted_query, sorted_query_ind) = unzip sorted_query_with_ind 

  let leafs_with_ind = zip indir leaves

  let knn_nat_leaf = map3 (\q q_ind l_ind -> bruteForce q init_knns[q_ind] 
                                              leafs_with_ind[scExc_leaves_shp[l_ind]:scInc_leaves_shp[l_ind]]
                          ) sorted_query sorted_query_ind sorted_query_leaf
			                      --queries (iota n) queries_init_leafs 

  let new_knns_sorted =
    loop (curr_nn_set) = (knn_nat_leaf) for i < (i64.i32 height) do
        let new_leaves = map (\l_num -> reverseBit l_num i) sorted_query_leaf 
                                                            --queries_init_leafs
        let better_nn_set = map3  (\q q_knn l_ind -> bruteForce q q_knn 
                                                      leafs_with_ind[scExc_leaves_shp[l_ind]:scInc_leaves_shp[l_ind]]
                                  ) sorted_query curr_nn_set new_leaves
					                          --queries curr_nn_set new_leaves
				
        in better_nn_set

  in scatter init_knns sorted_query_ind new_knns_sorted 
	      --(iota n) new_knns_sorted	


-- ToDos:
-- 1. Pass the desired `height` of the kd-tree as program parameter instead of `defppl`
-- 2. Fix the construction of the kd-tree to not split in the middle of a set of points
--      having the same value as the median.
--    Representation of kd-tree:
--      - You know that the tree is still fully balanced, but the number of points in
--        each leaf does not have to be the same.
--      - Therefore, when you are building the kd-tree, you need to keep track of the
--        shape of each level of the tree, i.e., level `i` has `2^i` nodes, hence it
--        corresponds to an arrray of `2^i` subarrays. The sum of the shape array has
--        to equal the number of reference points.
--      - on the rightside, you will probably pad with some empty arrays.
-- 3. adjust the rest of the code, i.e., of this file, to work with the new
--    kd-tree representation.
--
def main [m] [n] [d] (k: i64) (defppl: i32) (input: [m][d]f32) (queries: [n][d]f32) =
    let init_knns = replicate n (replicate k (-1i32, f32.inf))
    --- Build tree (height is without the leaf "level")
    let (height, num_inner_nodes, _) = computeTreeShape (i32.i64 m) defppl
    --let m'64 = i64.i32 m'
    let (leafs, indir, median_dims, median_vals, shape_arr) =
            mkKDtree height (i64.i32 num_inner_nodes) (m) input

    let leaves_shp = 
      let beg = i64.i32 (1 << (height + 1)) - 1
      let end = beg + (i64.i32 (1 << (height + 1)))
      in shape_arr[beg:end]
    let scInc_leaves_shp = scan (+) 0i32 leaves_shp |> map (i64.i32)
    let scExc_leaves_shp = [0i64] ++ (scInc_leaves_shp[:((length leaves_shp) - 1 )]) :> []i64

    -- 2. Find the leaf to which each query "naturally" belongs
    --    If your set of querries is named `querries` this is
    --    achieved with a map:
    let queries_init_leafs = map (findLeaf median_dims median_vals height) queries

    -- 3. sort `(zip querries leaf_inds)` in increasing order of
    --      their leaf_ind (second element)
    let (sorted_query_with_ind, sorted_query_leaf) = 
      let q_with_ind = zip queries (iota n) 
      let q_ind_with_leaf = zip q_with_ind queries_init_leafs
      in (radix_sort_int_by_key (\(_,l) -> l) (log2Int (length leafs)) i64.get_bit q_ind_with_leaf) |> unzip
    
    let (sorted_query, sorted_query_ind) = unzip sorted_query_with_ind 

    -- 4. for each querry compute its knns from its own leaf
    let leafs_with_ind = zip indir leafs

    -- Using values in sorted_query_leaf to index into scanned shape array.
    --   The values are used to "slice" the leaves correctly
    let knn_nat_leaf = map3 (\q q_ind l_ind -> bruteForce q init_knns[q_ind] 
                                                leafs_with_ind[scExc_leaves_shp[l_ind]:scInc_leaves_shp[l_ind]]
                            ) sorted_query sorted_query_ind sorted_query_leaf

    -- 5. have a loop which goes from [0 .. height - 1] which refines
    --    the nearest neighbors
    let new_knns_sorted =
      loop (curr_nn_set) = (knn_nat_leaf) for i < (i64.i32 height + 1) do
          let new_leaves = map (\l_num -> reverseBit l_num i) sorted_query_leaf
          let better_nn_set = map3  (\q q_knn l_ind -> bruteForce q q_knn 
                                                        leafs_with_ind[scExc_leaves_shp[l_ind]:scInc_leaves_shp[l_ind]]
                                    ) sorted_query curr_nn_set new_leaves
          in better_nn_set

    let new_knns = scatter (init_knns) sorted_query_ind new_knns_sorted 
    let (new_knn_ind, new_knn_dists) = unzip new_knns[0]

    in  (leafs, indir, median_vals, sorted_query_ind, new_knn_ind, new_knn_dists, leaves_shp)
