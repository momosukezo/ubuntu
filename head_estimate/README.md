# 顔検知の座標をcsvファイルに書き込み 10月9日更新

- NaN問題解決
- しかし，head_estimate_to_csv.pyは二人の顔を検知した場合のみ，csvファイルに書き込む
　※母と子のどちらかを特定することができないため，座標を見て判断するしかない

- nanachan_imgは画像，seinochan_video.pyは動画
- 顔検知して描画のみ
