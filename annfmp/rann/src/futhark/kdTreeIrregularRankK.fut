import "util"

-- m: the number of reference points
-- defppl: the default number of points per leaf
-- result: (height of tree without leaves, number of points per leaf)
def computeTreeShape (m: i32) (defppl: i32) : (i32, i32, i32) =
    let def_num_leaves = (m + defppl - 1) / defppl
    let hp1 = log2 def_num_leaves
    let h_2 = if def_num_leaves > (1 << hp1) then hp1 + 1 else hp1
    in
        if hp1 <= 0 then (-1, 0, m)
        else let h = h_2 - 1
             let num_leaves = 1 << (h+1)
             let ppl = (m + num_leaves - 1) / num_leaves
             in  (h, num_leaves-1, num_leaves*ppl)


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
def mkKDtree [m] [d] (height: i32) (q: i64) (m' : i64)
                     (input: [m][d]f32) :
           (*[m'][d]f32, *[m']i32, *[q]i32, *[q]f32, *[]i32) =

    let num_pads = m' - m
    let input' = input ++ (replicate num_pads (replicate d f32.inf)) :> [m'][d]f32
    let indir  = iota32 m'
    let total_nodes = q + (q + 1)
    
    let median_vals = replicate q 0.0f32
    let median_dims = replicate q (-1i32)
    let shape_arr   = replicate total_nodes (i32.i64 m')
    let ( indir' : *[m']i32
        , median_dims': *[q]i32
        , median_vals': *[q]f32
        , shape_arr':   *[total_nodes]i32 
        ) =
        loop ( indir  : *[m']i32
                , median_dims: *[q]i32
                , median_vals: *[q]f32
                , shape_arr:   *[total_nodes]i32)
                for lev < (height + 1) do

            let nodes_this_lvl = 1 << i64.i32 lev

            let start_shp = nodes_this_lvl - 1
            let end_shp   = start_shp + nodes_this_lvl
            let shp_this_lvl       = ( shape_arr[start_shp:end_shp]) :> [nodes_this_lvl]i32
            let scan_shp_this_lvl  = ( scan (+) 0 shp_this_lvl  )

            let med_dim = lev % (i32.i64 d) 

            -- Values of chosen dimension   rule 2 used (map inside map)
            let chosen_column = map (\ind -> input'[ind, med_dim]) indir 

            -- Calculate the median
            --   use rule 5 instead to calculate means (reduce inside map)
            let shp_flag_arr =  mkFlagArray shp_this_lvl (0i32) (replicate nodes_this_lvl 1i32) :> [m']i32
            let mins = sgmscan f32.min f32.highest shp_flag_arr chosen_column
            let maxs = sgmscan f32.max f32.lowest shp_flag_arr chosen_column
            let means = map (\ind -> if ind == 0 then 0
                                     else (mins[ind-1] + maxs[ind-1])/2.0f32 ) scan_shp_this_lvl 

            let ks  = map (\s -> s/2) shp_this_lvl
            -- let II1  = scan (+) 0i32 shp_flag_arr
            -- Correct computation of II1 in the presence of zero shape is:
            let flag_iotap1 = ( (indices shp_this_lvl)
                           |> map i32.i64
                           |> map (+1)
                           |> mkFlagArray shp_this_lvl (0i32)) :> [m']i32
            let II1 = sgmscan (+) 0 flag_iotap1 flag_iotap1
            let medians_this_lvl = rankSearchBatch means ks shp_this_lvl (copy II1) (copy chosen_column) 
                -- Got an error when not using copy. rankSearchBatch consumes II1 and chosen_colum

            --- Calculate the mask for partition3L
            ---   using ii1 to index into medians
            let mask_arr = map2 (\p_val ind -> p_val < medians_this_lvl[ind - 1]) chosen_column II1

            --- Partition to split each node
            let (indir'', new_splits) = partition3L2 mask_arr shp_flag_arr scan_shp_this_lvl (shp_this_lvl, indir) (0i32)

            --- For the shape I just need to 'weave' the shape with tmp below
            let tmp = map2 (\shp_val T_val -> shp_val - T_val ) shp_this_lvl new_splits
            let new_shape_ind = iota (nodes_this_lvl << 1)
            let new_shape = map (\ind -> if (ind % 2) == 0 then new_splits[ind/2] else tmp[ind/2]) new_shape_ind

            let this_lev_inds = map (+ (nodes_this_lvl-1)) (iota nodes_this_lvl)
            let next_lev_inds = map (+ ((nodes_this_lvl << 1)-1)) (iota (nodes_this_lvl << 1))
            let median_dims' = scatter median_dims this_lev_inds (replicate nodes_this_lvl med_dim)
            let median_vals' = scatter median_vals this_lev_inds medians_this_lvl
            let shape_arr''  = scatter (copy shape_arr) next_lev_inds new_shape --Consumes indir'' for some reason without copy
            in  (indir'', median_dims', median_vals', shape_arr'')

    let input'' = map (\ ind -> map (\k -> input'[ind, k]) (iota32 d) ) indir' :> *[m'][d]f32
    in  (input'', indir', median_dims', median_vals', shape_arr')



--def main3 [m] [d] (defppl: i32) (input: [m][d]f32) =
--    computeTreeShape (i32.i64 m) defppl
def buildKdTree [m] [d] (height: i32) (input: [m][d]f32) =
    let num_inner_nodes = (1 << height) - 1
    let (leaves, indir, median_dims, median_vals, shp_arr) =
        mkKDtree (height) (i64.i32 num_inner_nodes) m input
    in  (leaves, indir, median_dims , median_vals, shp_arr)
   

def main [m] [d] (defppl: i32) (input: [m][d]f32) =
    --let (height, num_inner_nodes, m') = computeTreeShape (i32.i64 m) defppl
    let h = defppl
    let nin = (1 << h) - 1
    let (leafs, indir, _, median_vals, shp_arr) =
        --mkKDtree height (i64.i32 num_inner_nodes) (m) input
        mkKDtree (h) (i64.i32 nin) (m) input
    in  (leafs, indir, median_vals, shp_arr, h, nin)
