# テンプレートマッチング(NCC)のプログラム
# 参照URL = https://algorithm.joho.info/programming/python/opencv-template-matching-ncc-py/
# 域値を設定することで、その域値以上の検出結果を描画するプログラム

# ver1に追記する
# 回転処理を加えたプログラムを記述する(達成)
# このプログラムは、回転しても全体がきちんと映るように調整されています。
# 

import cv2
import numpy as np

def template_matching_ncc(src, temp):
    # 画像の高さ・幅を取得
    h, w = src.shape
    ht, wt = temp.shape
    
    # スコア格納用の2次元リスト
    score = np.empty((h-ht, w-wt))

    # 配列のデータ型をuint8からfloatに変換
    src = np.array(src, dtype="float")
    temp = np.array(temp, dtype="float")

    # 走査
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # 窓画像
            roi = src[dy:dy + ht, dx:dx + wt]
            # NCCの計算式
            num = np.sum(roi * temp)
            den = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(temp ** 2)) 
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den            

    # スコアが最大(1に最も近い)の走査位置を返す
    #pt = np.unravel_index(score.argmax(), score.shape)
    #pt = np.unravel_index(np.where(score > 0.92), score.shape)
    pt = np.where(score > 0.87)
    #return (pt[1], pt[0])
    return pt

# テンプレートのスケールを変換する関数
def get_scale_template(scale,template):    
    scale = scale/10
    template = cv2.resize(template, dsize=(int(w_original*scale), int(h_original*scale)))
    return template

def get_rotate_template(temp,angle):
    # テンプレートのサイズを取得
    h, w = temp.shape    
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
        cv2.rectangle(img, (pt[1][i], pt[0][i]), (pt[1][i] + w, pt[0][i] + h), (0,0,200), 3)
    cv2.imwrite("./output/output2.png", img)



#=====================================#    
def main(): 
    # 入力画像とテンプレート画像をで取得
    img = cv2.imread("../../vm_part_img/IMG_1914_3.jpg")
    temp = cv2.imread("../../temp_img/IMG_1911.jpg")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)   

    # テンプレート画像の高さ・幅
    h, w = temp.shape

    # スケール変換実行
    
    # 回転変換実行
    angle = -10
    temp_rotate = get_rotate_template(temp,angle)
    cv2.imwrite("./output/output_rotate_{}.jpg".format(angle), temp_rotate)

    # アフィン変換実行
    
    # テンプレートマッチング（NumPyで実装）
    #pt = template_matching_ncc(gray, temp)

    # rgb画像に描画する
    #draw_line(pt,img)
    
#=====================================#    



if __name__ == "__main__":
    main()

