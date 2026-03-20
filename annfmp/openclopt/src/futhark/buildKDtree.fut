import "lib/github.com/diku-dk/sorts/radix_sort"
import "util"
import "rank-k-search"

local let closestLog2 (p: i32) : i32 =
    if p<=1 then 0
    else let (_,res) = loop (q,r) = (p,0)
                       while q > 1 do
                            (q >> 1, r+1)
         let err_down = p - (1 << res)
         let err_upwd = (1 << (res+1)) - p
         in  if err_down <= err_upwd
             then res else res+1

-- SHOULD PROBABLY BE REDEFINED TO GIVE TREE SHAPE FROM HEIGHT INSTEAD
-- m: the number of reference points
-- defppl: the default number of points per leaf
-- result: (height of tree without leaves, number of points per leaf)
let computeTreeShape (m: i32) (defppl: i32) : (i32, i32, i32, i32) =
    let def_num_leaves = (m + defppl - 1) / defppl
    let hp1 = closestLog2 def_num_leaves in
    if hp1 <= 0 then (-1, 0, m, m)
    else let h = hp1 - 1
         let num_leaves = 1 << (h+1)
         let ppl = (m + num_leaves - 1) / num_leaves
         in  (h, num_leaves-1, ppl, num_leaves*ppl)

local let updateBounds [n] [d2] (level: i32) (median_dims: [n]i32) (median_vals: [n]f32)
                     (node_ind: i32) (lubs_cur: *[d2]f32) : *[d2]f32=
    let d = d2 / 2
    let ancestor = 0
    let (_, res) =
      loop (ancestor,lubs_cur) for i < level do
        let k = level - i - 1
        let ancestor_child = compute_Kth_ancestor k node_ind
        let anc_dim = i64.i32 median_dims[ancestor]
        let lub_ind = if  (ancestor_child & 1) == 0 then anc_dim else d+anc_dim
                      -- if right node, then update lower bound
        let anc_med = median_vals[ancestor]
        let lubs_cur[lub_ind] = if !(f32.isinf anc_med) then anc_med else lubs_cur[lub_ind]
        in  (ancestor_child, lubs_cur)
    in  res

local let findClosestMed [n] (cur_dim: i32) (median_dims: [n]i32) (node_ind: i32) : i32 =
    let cur_node = node_ind
    let res_ind  = -1i32
    let (_, res) =
        loop (cur_node, res_ind)
          while (cur_node != 0) && (res_ind == (-1i32)) do
            let parent = getParent cur_node
            let res_ind = if median_dims[parent] == cur_dim then parent else -1
            in  (parent, res_ind)
    in  res

-- height: the height of the tree excluding leaves
-- q: the number of internal tree nodes (i.e., without leaves)
-- ppl: number of reference points per leaf
-- m' : the number of reference points hold in the tree
--    The following invariants hold: #leaves = q + 1 AND m' = (q+1) * ppl
-- input:  the d-dimensional array of reference points from which the tree is constructed
-- result: a tuple of five arrays
--         1. the reordered points (per leaf)
--         2. the indirect array that holds the original indices of each point
--         3. the index of the dimension that is split
--         4. the median value of the split dimension
--         5. the closest ancestor node index that splits the same dimension (or -1 if none)

-- mkKDtree takes a set of points, pads them to fit a complete KD-tree,
-- repeatedly sorts node point chunks by the widest dimension,
-- and returns both the reordered leaf layout and the per-node split metadata needed to represent the KD-tree.
let mkKDtree [m] [d] (height: i32) (q: i64)
                     (input: [m][d]f32) :
           (*[m][d]f32, *[1i64 << (i64.i32 height + 1i64)]i64, *[m]i32, *[q]i32, *[q]f32, *[q]i32) =

    -- We should know about shape in order to use rank k search
    -- We might need a variable here outside the loop, that can continuously be modified
    -- based on the current level of the tree.
    -- Maybe it should have length equal to the final amount of leaves and then just
    -- be sliced whenever it's given to rank k search function?

    -- Initial bounds used to calculate highest spread dimension.
    let inputT = transpose input
    let lbs = map (reduce_comm f32.min f32.highest) inputT |> opaque
    let ubs = map (reduce_comm f32.max f32.lowest ) inputT |> opaque
    let lubs = lbs ++ ubs

    -- Initializations
    let indir       = (map (\x -> i32.i64 x) (iota m)) -- should be m
    -- start value for shp is the full length of the input 
    let num_leaves  = 1i64 << (i64.i32 height + 1i64)
    let _ = trace num_leaves
    let shp0        = replicate num_leaves 0i64
    let shp         = scatter shp0 [0] [m]
    let median_vals = replicate q 0.0f32
    let median_dims = replicate q (-1i32)
    let clanc_eqdim = replicate q (-1i32)
    
    -- Loop
    let ( indir' : *[m]i32
        , shp' : *[num_leaves]i64
        , median_dims': *[q]i32
        , median_vals': *[q]f32
        , clanc_eqdim': *[q]i32
        ) =
    loop ( indir  : *[m]i32
        , shp : *[num_leaves]i64
        , median_dims: *[q]i32
        , median_vals: *[q]f32
        , clanc_eqdim: *[q]i32 )
        for lev < (height+1) do
            let nodes_this_lvl = 1i64 << i64.i32 lev
            let cur_shp = shp[0:nodes_this_lvl]

            -- compute the dimensions to be split for each node at this level
            -- and also the index of the closest ancestor that has split the
            -- same dimension
            -- For each node:
            -- compute current bounds by replaying all parent split choices to see what region the node lives in with updateBounds
            -- compute spread in each dimension
            -- choose the dimension with maximum spread
            -- find closest ancestor with same dimension using findClosestMed
            let (med_dims, anc_same_med) =
                map (\ (i: i32) ->
                        let node_ind = i + i32.i64 nodes_this_lvl - 1
                        -- walk from root to node and update bounds
                        let lubs_cur = updateBounds lev median_dims median_vals
                                                    node_ind (copy lubs)
                        -- chose dimension of highest spread
                        let diffs = map (\i -> f32.abs(lubs_cur[i+d] - lubs_cur[i])) (iota d)
                        let (cur_dim, _) = reduce_comm (\ (i1,v1) (i2,v2) ->
                                                            if v1 >= v2 then (i1, v1)
                                                                        else (i2, v2) )
                                                        (-1, f32.lowest) <| zip (map (\x -> i32.i64 x) (iota d)) diffs
                        let prev_anc = findClosestMed cur_dim median_dims node_ind
                        in  (cur_dim, prev_anc)
                    ) (map (\x -> i32.i64 x) (iota nodes_this_lvl))
                |> unzip
            
            ------------ RANK K SEARCH -------------
            -- This creates the segment ids for chosen_columns
            -- For instance shp = [1i64, 3i64] makes seg_ids become [0, 1, 1, 1]
            let offsets = exScan (+) 0 cur_shp                                           -- [0, 1]
            let flagArray = scatter (replicate m 0i64) offsets (map (\_ -> 1i64) cur_shp)    -- [1, 1, 0, 0]
            let II1 = scan (+) 0i64 flagArray                                                -- [1, 2, 2, 2]            
            let seg_ids = map (\x -> x - 1) II1                                              -- [0, 1, 1, 1]

            -- For each node chunk, grab only the coordinate values in the split dimension.
            -- So if a specific node splits on dimension 2, it extracts the 2nd coordinate of each point in that one node.
            let chosen_columns = map2 (\ind seg ->
                                        input[ind, med_dims[seg]]
                                    ) indir seg_ids

            let med_vals = computeMedianWithRankK cur_shp chosen_columns offsets flagArray II1
            
            --------- PARTITION2L -----------
            -- Hypothesis: VERIFIED
            -- Using indir instead of the input array in partition2L may allow us to track how each element has been scattered
            let bools = map2 (\x seg -> x < med_vals[seg]) chosen_columns seg_ids

            let indir = map i64.i32 indir
            let (splitInds, (_, indir')) = partition2L bools 0i64 (cur_shp, indir)
            let indir' = map i32.i64 indir'

            -- We now update the shape array
            -- Example:
            -- shp = [3, 5] -> shp' = [1, 2, 4, 1]
            -- splitInds = [1, 4]
            -- shp' = 1, 3-1, 4, 5-4
            let splitInds64 = map i64.i32 splitInds
            let _ = trace shp
            let cur_shp' = map2 (\len ind -> [ind] ++ [len - ind]) cur_shp splitInds64 |> flatten
            let shp' = scatter shp (iota (nodes_this_lvl * (1 + 1))) cur_shp'

            -- scatter the values of this level in the global result arrays
            let this_lev_inds = map (+ (nodes_this_lvl-1)) (iota nodes_this_lvl)
            let median_dims' = scatter median_dims this_lev_inds med_dims
            let median_vals' = scatter median_vals this_lev_inds med_vals[0:nodes_this_lvl]
            let clanc_eqdim' = scatter clanc_eqdim this_lev_inds anc_same_med
            -- let indir'' = flatten indir2d' :> *[m]i32

            in  (indir', shp', median_dims', median_vals', clanc_eqdim')

    let input' = map (\ ind -> map (\k -> input[ind, k]) (iota d) ) indir' :> *[m][d]f32
    in  (input', shp', indir', median_dims', median_vals', clanc_eqdim')


-- let main0 (m: i32) (defppl: i32) =
--     computeTreeShape m defppl

--   let defppl = 4i32
--   let input : [16][1]f32 =
--

-- ==
-- compiled input { 4i32 [[1f32],[2f32],[3f32],[4f32],[4f32],[5f32],[6f32],[7f32],[8f32],[8f32],[9f32],[2f32],[3f32],[4f32],[5f32]] }
-- -- output { 1 3 16 [[1f32],[2f32],[2f32],[3f32],[3f32],[4f32],[4f32],[4f32],[5f32],[5f32],[6f32],[7f32],[8f32],[8f32],[9f32],[f32.inf]] [0i32,1i32,11i32,2i32,12i32,3i32,4i32,13i32,5i32,14i32,6i32,7i32,8i32,9i32,10i32,15i32] [0, 0, 0] [4.5f32, 3f32, 7.5f32] [-1, 0, 0] }
-- let main [m][d] (defppl: i32) (input: [m][d]f32) =
--     let (height, num_inner_nodes, _ppl, m') = computeTreeShape (i32.i64 m) defppl
--     let (leafs, indir, median_dims, median_vals, clanc_eqdim) =
--           mkKDtree height (i64.i32 num_inner_nodes) (i64.i32 m') input
--     in  (height, num_inner_nodes, m', leafs, indir, median_dims, median_vals, clanc_eqdim)

--
-- Useful Commands
-- futhark pyopencl --library buildKDtree.fut
-- futhark bench --backend=opencl --pass-option=--device=AMD buildKDtree.fut
