def shiftPoints [n][d]
         (points : [n][d]f32) : [n][d]f32  =
         
     let mass_acc       = replicate (d) 0.0f32
     let mass_sum       = reduce (\acc point -> map2 (+) acc point ) mass_acc points
     let mass           = map (\i -> i / f32.i64 n) mass_sum
     let shifted_Points = map (\point -> map2 (\pv mv -> pv - mv) point mass) points
     in shifted_Points

def shiftTandQPoints [n] [m] [d]
         (test_points : [n][d]f32) (query_points : [m][d]f32) : ([n][d]f32, [m][d]f32)  =

     let mass_acc       = replicate (d) 0.0f32
     let mass_sum       = reduce (\acc point -> map2 (+) acc point ) mass_acc test_points
     let mass           = map (\i -> i / f32.i64 n) mass_sum
     let shifted_test_Points = map (\point -> map2 (\pv mv -> pv - mv) point mass) test_points
     let shifted_query_Points = map (\point -> map2 (\pv mv -> pv - mv) point mass) query_points
     in (shifted_test_Points, shifted_query_Points)


let main [n][d]
         (points : [n][d]f32)  =

    let mass_acc       = replicate (d) 0.0f32
    let mass_sum       = reduce (\acc point -> map2 (+) acc point ) mass_acc points
    let mass           = map (\i -> i / f32.i64 n) mass_sum
    let shifted_Points = map (\point -> map2 (\pv mv -> pv - mv) point mass) points
    in shifted_Points