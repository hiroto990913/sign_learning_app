## 手話学習アプリ

![スクリーンショット (15)](https://user-images.githubusercontent.com/67566912/173192765-5132ecad-6eb0-4119-b84c-130449576429.png)

## 概要
アルファベット手話学習コンテンツを作成しました。

## データ
Roboflow社が提供するPublic datasetsの「American Sign Language Letters Dataset」をダウンロードしました。
白黒にしたり反転したりする拡張機能がランダムに適応された画像をデータとして用いました。
https://public.roboflow.com/object-detection/american-sign-language-letters 

## モデルと精度について
モデルはImageNetで学習させたVGG16を手話の画像データを学習させたモデルを用いています。
精度は交差検証法でAccuracy Scoreを算出した結果0.8912と高い精度で認識できていることがわかりました。

## デモンストレーション
https://user-images.githubusercontent.com/67566912/173194196-b39dc758-1817-49bc-8cd9-0adfe84496c6.mp4

