let exScan 't [n]
  (op: t -> t -> t) (neutral: t) (xs: [n]t) : [n]t =
  let inc = scan op neutral xs
  in map (\i -> if i == 0 then neutral else inc[i-1]) (iota n)

import "util"
import "rank-k-search"
import "brute-force"
import "kd-traverse"
import "buildKDtree"
import "driverKNN"


def computeMedWithRankK [m] [n] (shp: [m]i32) (input: [n]f32)=
  let offsets = exScan (+) 0 (map i64.i32 shp)                                                   -- [0, 1]
  let flagArray = scatter (replicate n 0i64) offsets (map (\_ -> 1i64) shp)    -- [1, 1, 0, 0]
  let II1 = scan (+) 0i64 flagArray                                                -- [1, 2, 2, 2]

  -- Calculating mins
  let scanned_mins = sgmscan f32.min f32.highest (map i32.i64 flagArray) input
  let mins_inds = map2 (\off len -> off + len - 1) offsets (map i64.i32 shp)
  let mins = map (\i -> scanned_mins[i]) mins_inds

  -- Calculating maxs
  let scanned_maxs = sgmscan f32.max f32.lowest (map i32.i64 flagArray) input
  let maxs_inds = map2 (\off len -> off + len - 1) offsets (map i64.i32 shp)
  let maxs = map (\i -> scanned_maxs[i]) maxs_inds

  -- Calculating means
  let means = map2 (\min max -> (min + max) / 2) mins maxs --|> opague

  -- Calculate ks
  let ks = map (\ x -> i32.f32 (f32.floor ((f32.i32 x) / 2f32))) shp

  --
  let A = copy input
  let med_vals = rankSearchBatch means ks shp (map i32.i64 II1) A
  in med_vals

let main =

    -- let input = [23f32, 55f32, 72f32, 13f32, 16f32, 28f32, 65f32, 42f32, 16f32, 10f32, 13f32, 75f32, 92f32, 43f32, 66f32]
    -- let bob = map (\ point -> length point) input
    -- let flat = flatten input
    -- let _ = trace flat
    -- let _ = trace bob
    -- let bscan = exScan (+) 0 b
    -- let _ = trace bscan

    -- let condsL = [true,false,true,false,true,false,true,true,false,true,false,true,false,true,false,true]
    -- let dummy = 0i32
    -- let shp = [7, 9]
    -- let arr = [8i32,9i32,10i32,1i32,3i32,5i32,20i32,11i32,19i32,12i32,18i32,13i32,17i32,14i32,16i32,15i32]
    -- let (a,b) = partition2L condsL dummy (shp, arr)
    -- let _ = trace (a, b)
    -- let newshppart1 = map2 (\ x y -> x - y) (map i32.i64 shp) a
    -- let (x, y) = (zip a newshppart1)
    -- let (xx, xy) = x
    -- let (yx, yy) = y
    -- let flatnewshp = x ++ y
    -- let _ = trace (x, y)
    -- let _ = trace flatnewshp

    -- let _ = trace newshppart1
    -- let array = [[11f32, 1f32, 5f32, 20f32], [9f32, 54f32, 12f32, 2f32], [85f32, 59f32, 1857f32, 3f32], [66f32, 25f32, 17f32, 33f32]]
    -- let lengths = map length chosen_columns
    -- let ks = map (\ x -> ceil_div x 2) lengths
    -- -- number of points
    -- let N  = reduce (+) 0 lengths
    -- let _ = trace lengths
    -- let _ = trace ks
    -- let _ = trace N
    -- let total = 4
    -- let indir = iota total
    -- let shp = [1i64, 3i64]
    -- let offsets = exScan (+) 0 shp
    -- let start_flags = scatter (replicate total 0i64) offsets (map (\_ -> 1i64) shp)
    -- let seg_ids_1based = scan (+) 0i64 start_flags
    -- let seg_ids = map (\x -> x - 1) seg_ids_1based
    -- let med_dims = [0, 2]

    -- let chosen_columns = map2 (\ind seg ->
    --                                 array[ind, med_dims[seg]]
    --                                ) indir seg_ids
    -- let _ = trace chosen_columns

    -- let oneArray = map (\_ -> 1i32) shp

    -- let flagArray = mkFlagArray (map i32.i64 shp) 0 oneArray
    -- let scanned_mins = sgmscan f32.min f32.highest (map i32.i64 start_flags) chosen_columns
    -- let mins_inds = map2 (\off len -> off + len - 1) offsets shp
    -- let mins = map (\i -> scanned_mins[i]) mins_inds

    -- let scanned_maxs = sgmscan f32.max f32.lowest (map i32.i64 start_flags) chosen_columns
    -- let maxs_inds = map2 (\off len -> off + len - 1) offsets shp
    -- let maxs = map (\i -> scanned_maxs[i]) maxs_inds

    -- Calculating means
    -- let means = map2 (\min max -> (min + max) / 2) mins maxs --|> opague

    -- let _ = trace mins
    -- let _ = trace maxs
    -- let _ = trace means

    -- let ks = map (\ x -> i32.f32 (f32.floor ((f32.i32 x) / 2f32))) (map i32.i64 shp)
    -- let II1 = mkII1 (map i32.i64 shp)

    -- let _ = trace II1

    -- let A = copy chosen_columns
    -- let med_vals = rankSearchBatch means ks (map i32.i64 shp) II1 A
    -- let _ = trace med_vals
    -- let chosenbob = [11.0, 1857.0, 17.0, 12.0]

    -- let (medians, offsets, flagArray) = computeMedianWithRankK (map i32.i64 shp) chosenbob
    -- let _ = trace gimme
   -- offsets [0, 1]
   -- flagarray [1, 1, 0, 0]

    -- DO TOMORROW -> [11.0, 17.0, 17.0, 17.0] (maybe)
    -- make bool array
    -- test partition2L with indir
    -- scatter indir onto
    -- Compare all elements in chosenbob with their median using the shape OR using the offsets
    -- let bools = map2 (\flag element -> if flag == 1 then medInd++
    --                                    ) flagarray chosenbob
    -- let II1 = scan (+) 0i64 flagArray                                                -- [1, 2, 2, 2]
    -- let seg_ids = map (\x -> x - 1) II1
    -- let bools = map2 (\x seg -> x < medians[seg]) chosenbob seg_ids
    -- let _ = trace bools



    -- let indir = iota 4
    -- let gimme = partition2L bools 0i64 (shp, indir)
    -- let _ = trace gimme


    -- Update shp
    -- shp = [3, 5] -> shp' = [1, 2, 4, 1]
    -- splitInds = [1, 4]
    -- shp' = 1, 3-1, 4, 5-4
    -- shp' = zip

    -- let shp = [3i64, 5i64]
    -- let splitInds = [1i64, 4i64]
    -- let shp' = map2 (\len ind -> [ind] ++ [len - ind]) shp splitInds |> flatten
    -- let _ = trace shp'
    -- in shp'

    -- NaturalLeaves test:))
    let ref_pts = [[1.3f32, 4.7f32], [5.2f32, 7.9f32], [9.1f32, 3.6f32], [2.4f32, 8.8f32], [6.5f32, 1.2f32], [7.7f32, 5.5f32], [3.3f32, 9.9f32], [8.8f32, 2.1f32], [4.4f32, 6.6f32], [0.9f32, 3.3f32], [2.2f32, 7.7f32], [5.5f32, 8.1f32], [9.9f32, 0.4f32], [1.1f32, 2.8f32], [6.8f32, 4.2f32], [7.3f32, 9.0f32], [3.7f32, 1.5f32], [8.1f32, 6.3f32], [4.9f32, 2.2f32], [0.6f32, 7.4f32], [2.5f32, 5.8f32], [9.2f32, 8.6f32], [1.7f32, 3.9f32], [6.0f32, 4.5f32], [7.8f32, 0.7f32], [3.1f32, 9.4f32], [5.9f32, 6.2f32], [2.9f32, 5.1f32], [2.9f32, 5.1f32], [2.9f32, 5.1f32], [2.9f32, 5.1f32], [2.9f32, 5.1f32]]

    let median_dims = [1i32, 0i32, 0i32, 0i32, 1i32, 1i32, 0i32]
    let median_vals = [5.1f32, 6.0f32, 3.1f32, 1.7f32, 2.1f32, 5.1f32, 5.9f32]
    let queries = [[5f32, 6f32], [3f32, 9f32], [1f32, 4f32], [8f32, 6f32]]

    let shp = [3i64, 3i64, 3i64, 4i64, 0i64, 9i64, 5i64, 5i64]

    let (knn_inds, knn_vals, query_leaves0) = findNaturalLeaves 8i64 ref_pts shp median_dims median_vals queries

    let _ = trace knn_inds
    let _ = trace knn_vals
    let _ = trace query_leaves0

    in knn_vals

    -- let height = 2

    -- let num_inner_nodes = (1 << (height+1)) - 1
    -- let (leafs, shp, indir, median_dims, median_vals, _) =
    --       mkKDtree height (i64.i32 num_inner_nodes) ref_pts
    -- let (_, _, II1) = (computeOffsetsFlagsII1 32 (i64.i32 num_inner_nodes + 1) shp)
    -- let seg_ids = map (\x -> x - 1) II1
    -- let orig2leaf = scatter (replicate 32 (-1i32)) (map i64.i32 indir) (map i32.i64 seg_ids)

    -- let (knn_inds, knn_vals, nat_leaves) = findNaturalLeaves 8i64 leafs shp median_dims median_vals queries

    -- let knnss  = (map2 (\i v -> zip i v) knn_inds knn_vals)

    -- let knns = propagate leafs shp orig2leaf queries nat_leaves knnss

    -- let _ = trace knns

    -- in knns
