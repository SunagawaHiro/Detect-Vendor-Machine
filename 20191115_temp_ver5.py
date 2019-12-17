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
# OpenCVでマッチテンプレートを実行する
# OpecCVでマッチテンプレートを行うと、pythonのプログラムより早い。全体の処理が1分以内に完了する感じ
#

import cv2
import numpy as np


# テンプレートマッチング実行関数
def template_matching_ncc(src, temp):
    # 画像の高さ・幅を取得
    h, w = src.shape
    ht, wt = temp.shape
    
    # スコア格納用の2次元リスト
    score = np.empty((h-ht, w-wt))

    # 配列のデータ型をuint8からfloatに変換
    src = np.array(src, dtype="float")
    temp = np.array(temp, dtype="float")

    # 黒以外の部分でのマッチングをするようにアップデートする
  
    # 走査(これは元のプログラム)
    count = 0
    for dy in range(0, h - ht):
        count += 1
        for dx in range(0, w - wt):
            # 窓画像
            roi = src[dy:dy + ht, dx:dx + wt]                
            # NCCの計算式
            num = np.sum(roi * temp)
            den = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(temp ** 2))
            # おそらくここに原因がある
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den
            
        if count % 100 == 0:
            print("{}".format(count))

    # スコアが域値以上の走査位置(左上)を返す
    pt = np.where(score > 0.90)
    return pt


# テンプレートのスケールを変換する関数
def get_scale_template(scale,template):    
    scale = scale/10
    template = cv2.resize(template, dsize=(int(w_original*scale), int(h_original*scale)))
    return template

def get_rotate_template(temp,angle):
    # テンプレートのサイズを取得
    h, w = temp.shape[:2]    
    # 回転角の指定
    angle_rad = angle/180.0*np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

    temp_rot = cv2.warpAffine(temp, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    return temp_rot


# 描画する関数
def draw_line(pt, img):
    count = [len(v) for v in pt]
    for i in range(count[0]):
        cv2.rectangle(img, (pt[1][i], pt[0][i]), (pt[1][i] + 195, pt[0][i] + 101), (0,0,200), 3)
    cv2.imwrite("./output/output_input_rotate_result.png", img)



#=====================================#    
def main():
    FileName = "left"
    # 入力画像とテンプレート画像をで取得
    img_input = cv2.imread("../../vm_part_img/IMG_3188.jpg".format(FileName))
    temp = cv2.imread("../../temp_img/IMG_1911.jpg")

    # グレースケール変換
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)   

    # テンプレート画像の高さ・幅
    h, w = temp.shape    

    # スケール変換実行
    #scale = 1.0
    #scale_temp = get_scale_template(scale,temp)
    
    # 入力画像を回転させる
    angle = 10
    gray_rotate = get_rotate_template(gray_input,angle)
    #cv2.imwrite("./output/input_{}_{}.jpg".format(FileName,angle), gray_rotate)

    # テンプレート画像を回転させる
    temp_rotate = get_rotate_template(temp,angle)
    cv2.imwrite("./output/temp_{}_{}.jpg".format(FileName,angle), temp_rotate)

    # アフィン変換実行
    
    # テンプレートマッチング（NumPyで実装）
    #pt = template_matching_ncc(gray_input, temp)

    # テンプレートマッチング(OpenCV)
    res = cv2.matchTemplate(gray_input,temp_rotate,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
    #img_rotate = get_rotate_template(img_input,angle)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_input, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    # rgb画像に描画する
    #img_rotate = get_rotate_template(img_input,angle)
    cv2.imwrite("./output/{}_{}_full.jpg".format(angle,FileName), img_input)
    #draw_line(pt,img_rotate)
    
#=====================================#    



if __name__ == "__main__":
    main()

