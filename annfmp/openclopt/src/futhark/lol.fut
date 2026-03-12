let exScan 't [n]
  (op: t -> t -> t) (neutral: t) (xs: [n]t) : [n]t =
  let inc = scan op neutral xs
  in map (\i -> if i == 0 then neutral else inc[i-1]) (iota n)

import "util"

let main =

    -- let input = [23f32, 55f32, 72f32, 13f32, 16f32, 28f32, 65f32, 42f32, 16f32, 10f32, 13f32, 75f32, 92f32, 43f32, 66f32]
    -- let bob = map (\ point -> length point) input
    -- let flat = flatten input
    -- let _ = trace flat
    -- let _ = trace bob
    -- let bscan = exScan (+) 0 b
    -- let _ = trace bscan

    let condsL = [true,false,true,false,true,false,true,true,false,true,false,true,false,true,false,true]
    let dummy = 0i32
    let shp = [7, 9]
    let arr = [8i32,9i32,10i32,1i32,3i32,5i32,20i32,11i32,19i32,12i32,18i32,13i32,17i32,14i32,16i32,15i32]
    let (a,b) = partition2L condsL dummy (shp, arr)
    let _ = trace (a, b)
    let newshppart1 = map2 (\ x y -> x - y) (map i32.i64 shp) a
    let (x, y) = (zip a newshppart1)
    let (xx, xy) = x
    let (yx, yy) = y
    let flatnewshp = x ++ y
    let _ = trace (x, y)
    let _ = trace flatnewshp

    -- let _ = trace newshppart1

    in newshppart1

