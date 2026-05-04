let log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1

def iota32 n = (0..1..<i32.i64 n) :> [n]i32

def imap  as f = map f as
def imap2 as bs f = map2 f as bs   

def ones [q] 't (_xs: [q]t) = replicate q 1i32

let exScan 't [n]
  (op: t -> t -> t) (neutral: t) (xs: [n]t) : [n]t =
  let inc = scan op neutral xs
  in map (\i -> if i == 0 then neutral else inc[i-1]) (iota n)

def sgmscan 't [n] (op: t->t->t) (ne: t)
                   (flg : [n]i32) (arr : [n]t) : [n]t =
  let flgs_vals =
      scan ( \ (f1, x1) (f2,x2) ->
              let f = f1 | f2 in
              if f2 != 0 then (f, x2)
              else (f, op x1 x2) )
            (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

def irsgmscan 't [n] (op: t->t->t) (ne: t)
                    (flg : [n]bool) (arr : [n]t) : [n]t =
    let flgs_vals =
        zip flg arr |>
        scan (\ (f1,x1) (f2,x2) -> 
                let f = f1 || f2
                in if f2 then (f, x2)
                    else (f, op x1 x2)
                ) (false, ne)
    let (_, vals) = unzip flgs_vals
    in vals

-- Functions from DPP notes
def mkFlagArray 't [m]
            (aoa_shp: [m]i32) (zero: t)
            (aoa_val: [m]t  ) : []t =
  let shp_scn = scan (+) 0 aoa_shp
  let aoa_len = shp_scn[m-1]
  let shp_ind = imap2 aoa_shp (indices aoa_shp)
                      (\ s i ->
                         if s==0 then -1i64
                         else if i==0 then 0i64
                         else i64.i32 shp_scn[i-1]
                      )
  let flags = scatter (replicate (i64.i32 aoa_len) zero)
                      shp_ind aoa_val
  in flags

let mkII1 [m] (shp: [m]i32) : *[]i32 =
    let flags = mkFlagArray shp 0i8 (replicate m 1i8)
    in  map i32.i8 flags
     |> scan (+) 0i32

-- Function from DPP Notes Irregular 2dim Arrays
def mkIrFlagArray 't [m]
            (aoa_shp: [m]u32) (zero: t) 
            (aoa_val: [m]t) : ([m]u32, []t) =
    let shp_rot = map (\i-> if i == 0 then 0 else aoa_shp[i-1]) (iota m)
    let shp_scn = scan (+) 0 shp_rot
    let aoa_len = if m == 0 then 0i64 else i64.u32 <| shp_scn[m-1]+aoa_shp[m-1]
    let shp_ind = map2 (\shp ind -> if shp == 0 then -1i64 else i64.u32 ind) aoa_shp shp_scn
    let r = scatter (replicate aoa_len zero) shp_ind aoa_val
    in (shp_scn, r)


let partition2Ind [n] (cs: [n]bool) : ([n]i32, i32) =
    let tfs = map (\f -> if f then 1 else 0) cs
    let isT = scan (+) 0 tfs
    let ffs = map (\f -> if f then 0 else 1) cs
    let isF0 = scan (+) 0 ffs

    let i = isT[n-1]
    let isF = map (+ i) isF0
    let inds = map3 (\ c iT iF ->
                        if c then iT-1 else iF-1
                    ) cs isT isF
    let inds_gather = scatter isT (map (\x -> i64.i32 x) inds) (map (\x -> i32.i64 x) (iota n))
    in (inds_gather, i)

let sumSqrs [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    map2 (\x y -> let z = x-y in z*z) xs ys |> reduce (+) 0.0f32

let sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    loop (res) = (0.0f32) for (x,y) in (zip xs ys) do
        let z = x-y in res + z*z

let gather1D 't [m] (arr1D: []t) (inds: [m]i32) : *[m]t =
    map (\ind -> arr1D[ind] ) inds

let gather2D 't [m][d] (arr2D: [][d]t) (inds: [m]i32) : *[m][d]t =
    map (\ind -> map (\j -> arr2D[ind,j]) (iota d) ) inds

let scatter2D [m][k][n] 't (arr2D: *[m][k]t) (qinds: [n]i32) (vals2D: [n][k]t) : *[m][k]t =
  let nk = n*k
  let k' = i32.i64 k
  let flat_qinds = map (\i -> let (d,r) = (i / k', i % k')
                              in qinds[d]*k' + r
                       ) (map (\x -> i32.i64 x) (iota nk))
  let res1D = scatter (flatten arr2D) (map (\x -> i64.i32 x) flat_qinds) ((flatten vals2D) :> [nk]t)
  in  unflatten res1D

let getParent (node_index: i32) = (node_index-1) / 2

let isLeaf (h: i32) (node_index: i32) =
    node_index >= ((1 << (h+1)) - 1)


-- Please implement the function below, which is supposed to
-- be the lifted version of `partition2` function given above.
-- Arguments:
--   `(shp: [m]i32, arr: [n]t)` is the flat-representation of
--            the irregular 2-dim (input) array to be partitioned;
--            `shp` is its shape, and `arr` is its flat data;
--   `condsL` is an irregular 2-dim array of booleans, which has
--            the same shape (`shp`) and flat-length (`n`) as the
--            input to-be-partitioned array.
-- The result is a tuple:
--    the first element is an array of split points of size `m`,
--       i.e., the index in each segment where the `false` elements
--       start.
--    the second element is the flat-representation of the partitioned result:
--       the first element should simply be `shp` (redundant)
--       the second element should be the flat-data of the partitioned result.
let partition2L 't [n] [m]
               -- the shape of condsL is also shp
               (condsL: [n]bool) (dummy: t)
               (shp: [m]i64, arr: [n]t) :
               ([m]i32, ([m]i64, [n]t)) =
 let begs   = scan (+) 0 shp
 let flags  =  (  iota m
               |> map i32.i64
               |> map (+1)
               |> mkFlagArray (map i32.i64 shp) 0i32
               ) :> [n]i32

 let outinds= sgmscan (+) 0i32 flags <| (map (\f -> if f==0 then 0 else f-1) flags)

 let tflgsL = map (\c -> if c then 1i32 else 0i32) condsL
 let fflgsL = map (\b -> 1 - b) tflgsL

 let indsTL= sgmscan (+) 0i32 flags tflgsL
 let tmpL  = sgmscan (+) 0i32 flags fflgsL

 -- let lst = indsT[n-1]
 let lstL   = map2 (\s b -> if s==0 then -1 else #[unsafe] indsTL[b-1]
                   ) shp begs

 -- let indsF = map (+lst) tmp
 let indsFL = map2 (\t sgmind-> t + #[unsafe] lstL[sgmind]) tmpL outinds

 let indsL = map4(\c indT indF sgmind->
                       let offs = if sgmind > 0 then #[unsafe] begs[sgmind-1] else 0i64
                       in  if c then offs + (i64.i32 indT) - 1
                                else offs + (i64.i32 indF) - 1
                 ) condsL indsTL indsFL outinds

 let fltarrL = scatter (replicate n dummy) indsL arr
 in  (lstL, (shp,fltarrL))



-- the k'th ancestor of `node_ind` can be computed with
-- the formula: `(node_ind + 1 - (2^k)) / (2^k)`, for example
-- the parent           (k==1): `(node_ind - 1) / 2`
-- the grandparent      (k==2): `(node_ind - 3) / 4`
-- the grandgrandparent (k==3): `(node_ind - 7) / 8`
let compute_Kth_ancestor (k: i32) (node_ind: i32) =
    let tpk = 1 << k
    in  (node_ind + 1 - tpk) / tpk

let findNodeLevel (node: i32) : i32 =
 ( loop (lev, idx) = (0i32, node)
     while idx > 0i32 do
       (lev+1i32, getParent idx) ).0

-- given a tree `node1` at level `lev` and another tree leaf `leaf`,
-- this function computes the closest common ancestor of `node1` and `node2`
-- `h` is the height of the binary tree (without leaves).
let findClosestCommonAncestor (h: i32) (lev: i32) (node1: i32) (leaf: i32) : i32 =
    let node2 = compute_Kth_ancestor (h+1-lev) leaf
    let (res,_) =
      loop (node1, node2) while node1 != node2 do
        (getParent node1, getParent node2)
    in  res
