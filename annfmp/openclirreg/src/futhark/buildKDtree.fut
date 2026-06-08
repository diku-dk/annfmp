import "util"
import "rank-k-search"

-- Implemented DPP Notes on II1, FlagArray, Offset etc
let mkBFlagsII (size: i64) (nodes_this_lvl: i64) (shp: []i64) : ([]u32, []i64, []i64) =
    let (B1, F1) =  mkIrFlagArray (map u32.i64 shp) 0 (map (\x -> x+1) (iota nodes_this_lvl))
    let F1fix = F1 :> [size]i64
    let Farr = map bool.i64 F1
    let II1 = segmented_scan (+) 0 Farr F1 :> [size]i64
    in (B1, F1fix, II1)

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
        let lubs_cur[lub_ind] = median_vals[ancestor]
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
-- input:  the d-dimensional array of reference points from which the tree is constructed
-- result: a tuple of six arrays
--         1. the reordered points (per leaf)
--         2. the shape array that contains the size of each leaf
--         3. the indirect array that holds the original indices of each point
--         4. the index of the dimension that is split
--         5. the median value of the split dimension
--         6. the closest ancestor node index that splits the same dimension (or -1 if none)

let mkKDtree [m] [d] (height: i32) (q: i64)
                     (input: [m][d]f32) :
           (*[m][d]f32, *[q+1]i64, *[m]i32, *[q]i32, *[q]f32, *[q]i32) =

    -- Initial bounds used to calculate highest spread dimension.
    let inputT = transpose input
    let lbs = map (reduce_comm f32.min f32.highest) inputT |> opaque
    let ubs = map (reduce_comm f32.max f32.lowest ) inputT |> opaque
    let lubs = lbs ++ ubs

    -- Initializations
    let indir       = (map (\x -> i32.i64 x) (iota m))
    -- start value for shp is the full length of the input
    let shp         = scatter (replicate (q+1) 0i64) [0] [m]
    let median_vals = replicate q 0.0f32
    let median_dims = replicate q (-1i32)
    let clanc_eqdim = replicate q (-1i32)

    -- Loop carried variables
    let ( indir' : *[m]i32
        , shp' : *[q+1]i64
        , median_dims': *[q]i32
        , median_vals': *[q]f32
        , clanc_eqdim': *[q]i32
        ) =
    loop ( indir  : *[m]i32
        , shp : *[q+1]i64
        , median_dims: *[q]i32
        , median_vals: *[q]f32
        , clanc_eqdim: *[q]i32 )
        for lev < (height+1) do
            let nodes_this_lvl = 1i64 << i64.i32 lev
            let cur_shp = shp[0:nodes_this_lvl]
            let (med_dims, anc_same_med) =
                map (\ (i: i32) ->
                        let node_ind = i + i32.i64 nodes_this_lvl - 1
                        -- walk from root to node and update bounds
                        let lubs_cur = updateBounds lev median_dims median_vals
                                                    node_ind (copy lubs)
                        let _ = trace lubs_cur
                        -- chose dimension of highest spread
                        let diffs = map (\i -> f32.abs(lubs_cur[i+d] - lubs_cur[i])) (iota d)
                        let _ = trace diffs
                        let (cur_dim, _) = reduce_comm (\ (i1,v1) (i2,v2) ->
                                                            if v1 >= v2 then (i1, v1)
                                                                        else (i2, v2) )
                                                        (-1, f32.lowest) <| zip (map (\x -> i32.i64 x) (iota d)) diffs
                        let _ = trace cur_dim
                        let prev_anc = findClosestMed cur_dim median_dims node_ind
                        in  (cur_dim, prev_anc)
                    ) (map (\x -> i32.i64 x) (iota nodes_this_lvl))
                |> unzip

            ------------ RANK K SEARCH -------------
            let (offsetInds, flags, II1) = mkBFlagsII m nodes_this_lvl cur_shp
            let seg_ids = map (\x -> x - 1) II1

            -- For each node chunk, grab only the coordinate values in the split dimension.
            -- So if a specific node splits on dimension 2, it extracts the 2nd coordinate of each point in that one node.
            let chosen_columns = map2 (\ind seg ->
                                        input[ind, med_dims[seg]]
                                    ) indir seg_ids

            let med_vals = computeMedianWithRankK cur_shp chosen_columns (map i64.u32 offsetInds) flags II1

            --------- PARTITION2L -----------
            -- Split nodes by < and >= predicates on values
            let bools = map2 (\x seg -> x < med_vals[seg]) chosen_columns seg_ids

            let indir = map i64.i32 indir
            let (splitInds, (_, indir')) = partition2L bools 0i64 (cur_shp, indir)
            let indir' = map i32.i64 indir'
            let splitInds64 = map i64.i32 splitInds

            -- We now update the shape array
            -- Example:
            -- shp = [3, 5] -> shp' = [1, 2, 4, 1]
            -- splitInds = [1, 4]
            -- shp' = [1, 3-1, 4, 5-4]
            -- if splitInd == -1 then [0, 0]
            let cur_shp' = map2 (\len ind -> if len == 0 then [len] ++ [len] else [ind] ++ [len - ind]) cur_shp splitInds64 |> flatten
            let shp' = scatter shp (iota (nodes_this_lvl * (1 + 1))) cur_shp'

            -- scatter the values of this level in the global result arrays
            let this_lev_inds = map (+ (nodes_this_lvl-1)) (iota nodes_this_lvl)
            let median_dims' = scatter median_dims this_lev_inds med_dims
            let median_vals' = scatter median_vals this_lev_inds med_vals[0:nodes_this_lvl]
            let clanc_eqdim' = scatter clanc_eqdim this_lev_inds anc_same_med

            in  (indir', shp', median_dims', median_vals', clanc_eqdim')

    let input' = map (\ ind -> map (\k -> input[ind, k]) (iota d) ) indir' :> *[m][d]f32
    in  (input', shp', indir', median_dims', median_vals', clanc_eqdim')
