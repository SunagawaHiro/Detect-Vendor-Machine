# テンプレートマッチング(NCC)のプログラム
# 参照URL = https://algorithm.joho.info/programming/python/opencv-template-matching-ncc-py/
# 域値を設定することで、その域値以上の検出結果を描画するプログラム

# ver1に追記する
# 回転処理を加えたプログラムを記述する(達成)
# このプログラムは、回転しても全体がきちんと映るように調整されています。

# ver2に追記
# 回転画像の黒以外の部分をテンプレートマッチングにかける(未達成)
# Q. 入力画像を回転させ、それに対してテンプレートマッチングを試すことで解決することはできないだろうか？
# A. できた。pythonとOpenCVの関数のそれぞれでマッチングした時の処理結果の違いが少し気になる

# ver4に追記
# OpenCVでマッチテンプレートを実行する処理を追加した
# OpecCVでマッチテンプレートを行うと、pythonのプログラムより早い。全体の処理が1分以内に完了する感じ

# ver5に追記
# 正面画像に限定する方針に切り替え
# python のマッチテンプレート関数は削除しました
# 回転させる処理も削除しました
# 回転を行う関数も削除しました
# 描画する関数も削除しました
# カラーヒストグラムを行う関数を追加しました(参照URL https://qiita.com/best_not_best/items/c9497ffb5240622ede01)
# カラーヒストグラムの類似度の結果を表示できるようになりました
# 3511.jpg


import cv2
import numpy as np

# テンプレートのスケールを変換する関数
def get_scale_template(scale,template):    
    scale = scale/10
    # 今は応急処置として,手動でテンプレートのサイズを設定する
    w_original = 195 
    h_original = 101 
    template = cv2.resize(template, dsize=(int(w_original*scale), int(h_original*scale)))
    return template

# カラーヒストグラムを用いた類似度マッチング
def color_hist_compare(temp_img, crop_img):
    
    # リサイズする関数(スケール変更した際に必要になりそう)
    IMG_SIZE = (195,101)
    temp_img = cv2.resize(temp_img, IMG_SIZE)
    cv2.imwrite("./output/temp.png",temp_img)
    crop_img = cv2.resize(crop_img, IMG_SIZE)
    cv2.imwrite("./output/hikaku_taisyou.png",crop_img)

    # テンプレートとクロップ画像のヒストグラムを計算し, 比較する
    target_hist = cv2.calcHist([temp_img], [0], None, [256], [0, 256])    
    comparing_hist = cv2.calcHist([crop_img], [0], None, [256], [0, 256])        
    ret = cv2.compareHist(target_hist, comparing_hist, 0)

    return ret

    



def main(): #=====================================
    FileName = "3511"
    # 入力画像とテンプレート画像をで取得
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    #img_input = cv2.imread("../../vm_1911_same_button/IMG_3514.jpg")
    temp = cv2.imread("../../temp_img/IMG_1911.jpg")

    # グレースケール変換
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)   

    # テンプレート画像の高さ・幅
    h, w = temp.shape    

    # スケール変換実行
    scale = 14
    temp = get_scale_template(scale,temp)    

    ht,wt = temp.shape
    print(ht,wt)
        
    # テンプレートマッチング(OpenCV)
    res = cv2.matchTemplate(gray_input,temp,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_input, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)    
    # rgb画像に域値以上の結果を描画する
    cv2.imwrite("./output/ikiti_ijou_{}_{}_{}.jpg".format(scale,FileName,threshold), img_input)

    # rgb画像にmaxの類似値を持つ結果を出力する
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    #img_input = cv2.imread("../../vm_1911_same_button/IMG_3514.jpg")
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(res)
    print(max_value)
    pt = max_pt
    # 類似度最大の場所に対して描画する
    #cv2.rectangle(img_input, (pt[0], pt[1] ), (pt[0] + wt, pt[1] + ht), (0,0,200), 3)
    #cv2.imwrite("./output/whre_max_pt.png", img_input)


    # 類似度が最大の出力結果をクロップする
    similar_max_img = img_input[ pt[1] : pt[1]+ht, pt[0] : pt[0]+wt] # img[top : bottom, left : right]
    cv2.imwrite("./output/max_crop.png", similar_max_img)

    # 適当な画像で検証してみる
    #similar_max_img_t = img_input[ 644 : 794, 871 : 1159] 

    # ヒストグラム比較を行う
    ret = color_hist_compare(temp, similar_max_img)
    print("類似度:{}".format(ret))
    
#=====================================#    



if __name__ == "__main__":
    main()

