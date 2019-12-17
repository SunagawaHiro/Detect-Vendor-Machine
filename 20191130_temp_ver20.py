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
# 正面画像に限定する方針に切り替えました。それにより,回転処理系の関数などは消すようになっています。
# python のマッチテンプレート関数は削除しました
# 回転させる処理も削除しました
# 回転を行う関数も削除しました
# 描画する関数も削除しました
# カラーヒストグラムを行う関数を追加しました(参照URL https://qiita.com/best_not_best/items/c9497ffb5240622ede01)
# カラーヒストグラムの類似度の結果を表示できるようになりました
# 3511.jpg

# ver6に追記
# 画像の2点を指定し, その直線上のヒストグラムを求める関数を追加しました

# ver7に追記
# グラフのパターンを検出したい

# ver8に追記
# 自己相関関数を使えるようにしました

# ver9に追記
# 2点間を通る直線の方程式(y=m*x+n)がわかるようになりました
# 類似度グラフを表示できるようになりました(座標は画像の座標と対応していません)

# ver10に追記
# 類似度グラフを表示できるようになりました(グラフの横軸座標と画像座標は対応しています)
# 3511の類似度データを他ファイルから読み込めるようになりました. マッチングの計算を行わなくてもよくなりました.
# ピーク検出を行なった類似度グラフを表示できるようになりました.

# ver11に追記
# 4245.jpg の画像に対してこのプログラムを適用しました

# ver12に追記
# 3916.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.
# 3841.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.

# ver13に追記
# テンプレートマッチングで検出した個数をグラフにして表示できるようになりました.
# グラフのピーク値から, どこにボタン列があるのかを判定できるようになりました.

# ver14に追記
# 検出ボタンの個数グラフのピーク値(どの座標にボタン列がありそうかが分かる)から,
# 各列における類似度の高いボタンを4つ抜き出すことができるようになりました.
# 検出したボタン列が3列以外(0,1,2列)の場合, プログラムがうまくいかないバグが発見されました.

# ver15に追記
# ver15で発見されたバグを直しました.すなわちピーク(ボタン列)が3列以外だとしても動作するようになりました
# 

# ver16に追記
# スケール変換しながらテンプレマッチングを行なった際,最も検出個数が多かったときのテンプレートのスケール,
# そのスケールで変換を行なった後のテンプレートを取得できるようになりました.
# スケール変換の関数を削除しました.
# ボタン列のピーク値が本当に正しいかどうかを判定する機能を追加しました.

# ver17に追記
# 検出された1つのボタン列のピークについて,回帰直線を引いて, その間のパターンを求めることができました。

# ver18に追記
# 検出された全てのボタン列に対して回帰直線を引いて, ピーク検出を行えるようになりました.

# ver19に追記
# 類似度が域値以上のピークのみを抽出することができるようになりました.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as smg
from statsmodels.graphics.tsaplots import plot_acf
import match_data as md # 独自モジュール


# カラーヒストグラムを用いた類似度マッチング
def color_hist_compare(temp_img, crop_img):
    FileName = "3841"
    
    # リサイズする関数(スケール変更した際に必要になりそう)
    IMG_SIZE = (191,102)
    temp_img = cv2.resize(temp_img, IMG_SIZE)
    cv2.imwrite("./output/{}_jpg/temp.png".format(FileName), temp_img)
    crop_img = cv2.resize(crop_img, IMG_SIZE)
    cv2.imwrite("./output/{}_jpg/hikaku_taisyou.png".format(FileName), crop_img)

    # テンプレートとクロップ画像のヒストグラムを計算し, 比較する
    target_hist = cv2.calcHist([temp_img], [0], None, [256], [0, 256])    
    comparing_hist = cv2.calcHist([crop_img], [0], None, [256], [0, 256])        
    ret = cv2.compareHist(target_hist, comparing_hist, 0)

    return ret



# 2点間の直線のヒストグラムを求める関数(まだ効果はわからない。これから使えるかどうか検証する)
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]
    
    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    
    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)
    
    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
    
    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


# 移動平均を行う関数
def get_mave(sum_list, radius):
    mave_sum_x = []
    for i in range(radius, len(sum_list)-radius):
        sum_element = 0
        sum_count = 0
        for j in range(-radius, radius+1, 1):
            sum_element += sum_list[i+j]
            sum_count += 1
        mave_sum_x.append(sum_element/sum_count)        
    return mave_sum_x

# 自己相関を求める関数
def autocorrelation(data, k):
    """Returns the autocorrelation of the *k*th lag in a time series data.

    Parameters
    ----------
    data : one dimentional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """

    # yの平均
    y_avg = np.mean(data)

    # 分子の計算
    sum_of_covariance = 0
    for i in range(k+1, len(data)):
        covariance = ( data[i] - y_avg ) * ( data[i-(k+1)] - y_avg )
        sum_of_covariance += covariance

    # 分母の計算
    sum_of_denominator = 0
    for u in range(len(data)):
        denominator = ( data[u] - y_avg )**2
        sum_of_denominator += denominator

    return sum_of_covariance / sum_of_denominator



# 2点を通る直線の方程式を返す(y=mx+n)
def makeLinearEquation(x1, y1, x2, y2):
	line = {}
	if y1 == y2:
		# y軸に平行な直線
		line["y"] = y1
	elif x1 == x2:
		# x軸に平行な直線
		line["x"] = x1
	else:
		# y = mx + n
		line["m"] = (y1 - y2) / (x1 - x2)
		line["n"] = y1 - (line["m"] * x1)
	return line

# x座標を渡すと, y座標が返ってくる関数
def get_y_crd(m,n,x):
    y = m*x + n
    return y

def get_button_line_points3(s_x_y, maxid, ht, wt):
    P1 = []
    P2 = []
    P3 = []    
    for i in s_x_y:
        # P1について
        if (maxid[0][0] - ht <= i[2] <= maxid[0][0] + ht) and len(P1) < 4: # y座標が1段目(ピーク1)の範囲にある時
            if len(P1) == 0:  # P1が空ならP1にそのまま代入する
                P1.append(i)
            elif len(P1) == 1: # P1に1つ値が格納されているならば
                if abs(P1[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P1.append(i)
            elif len(P1) == 2:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P1.append(i)
            elif len(P1) == 3:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5 and abs(P1[2][1] - i[1]) >= wt*0.5:
                    P1.append(i)                    
        # P2について
        elif (maxid[0][1] - ht <= i[2] <= maxid[0][1] + ht) and len(P2) < 4: # y座標が2段目(ピーク1)の範囲にある時
            if len(P2) == 0:  # P2が空ならP2にそのまま代入する
                P2.append(i)
            elif len(P2) == 1: # P2に1つ値が格納されているならば
                if abs(P2[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P2.append(i)
            elif len(P2) == 2:
                if abs(P2[0][1] - i[1]) >= wt*0.5 and abs(P2[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P2.append(i)
            elif len(P2) == 3:
                if abs(P2[0][1] - i[1]) >= wt*0.5 and abs(P2[1][1] - i[1]) >= wt*0.5 and abs(P2[2][1] - i[1]) >= wt*0.5:
                    P2.append(i)                    
        # P3について
        elif (maxid[0][2] - ht <= i[2] <= maxid[0][2] + ht) and len(P3) < 4: # y座標が1段目(ピーク1)の範囲にある時
            if len(P3) == 0:  # P3が空ならP3にそのまま代入する
                P3.append(i)
            elif len(P3) == 1: # P3に1つ値が格納されているならば
                if abs(P3[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P3.append(i)
            elif len(P3) == 2:
                if abs(P3[0][1] - i[1]) >= wt*0.5 and abs(P3[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P3.append(i)
            elif len(P3) == 3:
                if abs(P3[0][1] - i[1]) >= wt*0.5 and abs(P3[1][1] - i[1]) >= wt*0.5 and abs(P3[2][1] - i[1]) >= wt*0.5:
                    P3.append(i)                            
        if len(P1)+len(P2)+len(P3) >= 12:
                break        
    return P1,P2,P3


def get_button_line_points2(s_x_y, maxid, ht, wt):
    P1 = []
    P2 = []    
    for i in s_x_y:
        # P1について
        if (maxid[0][0] - ht <= i[2] <= maxid[0][0] + ht) and len(P1) < 4: # y座標が1段目(ピーク1)の範囲にある時
            if len(P1) == 0:  # P1が空ならP1にそのまま代入する
                P1.append(i)
            elif len(P1) == 1: # P1に1つ値が格納されているならば
                if abs(P1[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P1.append(i)
            elif len(P1) == 2:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P1.append(i)
            elif len(P1) == 3:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5 and abs(P1[2][1] - i[1]) >= wt*0.5:
                    P1.append(i)                    
        # P2について
        elif (maxid[0][1] - ht <= i[2] <= maxid[0][1] + ht) and len(P2) < 4: # y座標が2段目(ピーク1)の範囲にある時
            if len(P2) == 0:  # P2が空ならP2にそのまま代入する
                P2.append(i)
            elif len(P2) == 1: # P2に1つ値が格納されているならば
                if abs(P2[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P2.append(i)
            elif len(P2) == 2:
                if abs(P2[0][1] - i[1]) >= wt*0.5 and abs(P2[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P2.append(i)
            elif len(P2) == 3:
                if abs(P2[0][1] - i[1]) >= wt*0.5 and abs(P2[1][1] - i[1]) >= wt*0.5 and abs(P2[2][1] - i[1]) >= wt*0.5:
                    P2.append(i)                                               
        if len(P1)+len(P2) >= 8:
                break        
    return P1,P2


def get_button_line_points1(s_x_y, maxid, ht, wt):
    P1 = []
    for i in s_x_y:
        # P1について
        if (maxid[0][0] - ht <= i[2] <= maxid[0][0] + ht) and len(P1) < 4: # y座標が1段目(ピーク1)の範囲にある時
            if len(P1) == 0:  # P1が空ならP1にそのまま代入する
                P1.append(i)
            elif len(P1) == 1: # P1に1つ値が格納されているならば
                if abs(P1[0][1] - i[1]) >= wt*0.5: # 格納されている1つの値のx座標と比較して, 離れていれば
                    P1.append(i)
            elif len(P1) == 2:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5: # 格納されている2つの値のx座標と比較
                    P1.append(i)
            elif len(P1) == 3:
                if abs(P1[0][1] - i[1]) >= wt*0.5 and abs(P1[1][1] - i[1]) >= wt*0.5 and abs(P1[2][1] - i[1]) >= wt*0.5:
                    P1.append(i)                            
        if len(P1) >= 4:
                break        
    return P1

# 最も検出個数の多いスケールとそのときの画像を取得する
def get_max_button_scale(gray_input,temp):    
    nums_scale = []
    h_original, w_original = temp.shape
    threshold = 0.7
    button_nums = []
    for i in range(5,16,1):
        scale = i/10
        template = cv2.resize(temp, dsize=(int(w_original*scale), int(h_original*scale)))
        res = cv2.matchTemplate(gray_input,template,cv2.TM_CCOEFF_NORMED)    
        loc = np.where( res >= threshold)
        button_nums.append(len(loc[0]))
        
    print("button_nums = ",button_nums)
    print("max(button_nums):",max(button_nums))
    print("button_nums.index(max(button_nums))",button_nums.index(max(button_nums)))
    nums_scale.append(max(button_nums))
    nums_scale.append(button_nums.index(max(button_nums))+5)
        
    scale = nums_scale[1]/10
    template = cv2.resize(temp, dsize=(int(w_original*scale), int(h_original*scale)))
    
    return nums_scale,template

def get_xy_points(P1, w_move, h_move):
    x_points = []
    y_points = []    
    for i in P1:
        x_points.append(i[1]+w_move)
        y_points.append(i[2]+h_move)
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    return x_points,y_points

def get_new_line_crd(func,w_input,w_move,h_move):
    line_crd = []
    count = 0
    for x in range(0,w_input,1):
        value = (x,int(func(x)))
        line_crd.append(value)    
        count += 1
        if (count % 200) == 0:
            print("value = ",value)
            
    print("len(line_crd) = ",len(line_crd))
    print()    
    # 左上の座標に持って行くための移動幅(line_crdをテンプレの中心として、テンプレートを切り抜かなければならない。)
    # そのため, line_crd に格納されている座標から左上に持っていくために テンプレのサイズ(縦横)の半分を知っている必要がある
    print("h_move",h_move)
    print("w_move",w_move)
    # 直線の左端と右端を削除する(テンプレートマッチングとして利用しないため)
    print("line_crd の長さ",len(line_crd))
    new_line_crd = line_crd[w_move:w_input-w_move]
    print("new_line_crd の長さ",len(new_line_crd))
    return new_line_crd


def get_ruijido_list(new_line_crd,FileName,w_move,h_move,wt,ht,temp):
    # クロップして,matchテンプレート関数にかける
    # 直線上の類似度マップを求める. +のx軸方向に1つずつ座標をずらしながら類似度を計算していく
    ruijido_list = [] # 類似度を格納する配列
    count = 0
    for i in range(len(new_line_crd)):        
        gray_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName),0)
        top = new_line_crd[i][1] - h_move
        bottom = new_line_crd[i][1] + h_move
        left = new_line_crd[i][0] - w_move
        right = new_line_crd[i][0] + w_move
        crop_each = gray_input[ top : bottom, left : right] # img[top : bottom, left : right]        
        # リサイズする
        crop_each = cv2.resize(crop_each , (wt,ht))        
        # テンプレートマッチングを行う
        res = cv2.matchTemplate(crop_each,temp,cv2.TM_CCOEFF_NORMED)        
        ruijido_list.append(res[0][0])
        count += 1        
        if count % 200 == 0:
            print(count)            
    return ruijido_list


def draw_graph(ruijido_list,w_move,w_input,FileName,count):            
    left = np.array(list(range(w_move,w_input-w_move,1)))

    ruijido_list = np.array(ruijido_list)
    print("len(ruijido_list)",len(ruijido_list))
    
    plt.bar(left, ruijido_list)
    plt.savefig("./output/{}_jpg/20191129/{}_ruijido_graph_S{}.png".format(FileName,FileName,count))    
    print("{}枚目".format(count))
    plt.close()

    
    
def main(): #=====================================
    FileName = "3841"
    # 入力画像とテンプレート画像をで取得
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    temp = cv2.imread("../../temp_img/IMG_2159.jpg")    

    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # 入力画像の高さ・幅
    h_input,w_input = gray_input.shape
    print(h_input,w_input)

    # テンプレート画像の高さ・幅
    h, w = temp.shape
    print("元のテンプレの高さ,幅:",h,w)

    # 検出個数が最も多いときのスケールを取得する
    max_btn_nums_scl,temp = get_max_button_scale(gray_input,temp) 
    print("最も検出されたときの個数,スケール",max_btn_nums_scl)    
    ht,wt = temp.shape
    print("スケール変換後のテンプレの高さ・幅",ht,wt)
            
    # テンプレートマッチング(OpenCV)
    res = cv2.matchTemplate(gray_input,temp,cv2.TM_CCOEFF_NORMED)
    #print("res = ",res)
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_input, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)
    print("検出個数:",len(loc[0]))
    print()
        
    # rgb画像に域値以上の結果を描画する
    cv2.imwrite("./output/{}_jpg/ikiti_ijou_{}_{}_{}.jpg".format(FileName,max_btn_nums_scl[1],FileName,threshold), img_input)

    # 縦軸(検出個数), 横軸(画像の縦サイズ) の類似度グラフを作成する ========================
    left = list(range(0,h_input,1))
    tate_graph = [0]*h_input                

    # 左上のy座標ではなく,ボタンの中心座標を格納するよう変更
    w_move = int(wt/2)
    h_move = int(ht/2)
    for i in loc[0]:     
        tate_graph[i+h_move] += 1

    # グラフを保存する
    #plt.bar(left, tate_graph)    
    #plt.savefig("./output/3841_jpg/num_button_graph/button_num_detect_add_hmove.png")

    # ピークを検出する
    data_3841_numpy = np.array(tate_graph)
    peak = wt    
    maxid = signal.argrelmax(data_3841_numpy, order=peak)
    print("ピーク位置:",maxid[0])
    
            
    # ピークが使えるものかどうかを判定し, 新たな配列に確実なピークを格納する
    new_maxid = []
    if 1 <= len(maxid[0]) <= 3: # ピークが1つ以上3つ以下なら
        for i in maxid[0]:
            a = tate_graph[i-2:i+3] # ピーク周辺の合計を計算
            print("a = ",a)
            print("sum(a) = ",sum(a))
            if sum(a) >= len(loc[0])/20: # ピーク周辺の合計検出数が, 全体検出数の5%以上なら,新しい配列にピーク値代入
                new_maxid.append(i)
                
    # maxidのshapeのままで情報をnew_maxid_2に移植する(この形にしないと,後の類似度の高いボタンを検出するコードが働かない)
    # この辺りは,スパゲティコードなのでアルゴリズムなどは考え直した方が良い
    print("new_max_id = ",new_maxid)
    print()    
    new_maxid_2 = []
    new_maxid_2.append(new_maxid)
    new_maxid_2.append(new_maxid)
    print("new_maxid_2 = ",new_maxid_2)
            
    # ピークのグラフを描画する
    Fig = plt.figure()
    Map1 = Fig.add_subplot(111)    
    Map1.plot(left, data_3841_numpy)
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')        
    #plt.savefig("./output/{}_jpg/num_button_graph/{}_midstep_peak_add_hmove.png".format(FileName, FileName))
    plt.close()
    
    # ====== ピークの位置から, 各ボタン列の中にある類似度の高いボタンを3つ選び出す =========================

    # 域値以上の類似度を格納する
    similar_to_loc = []
    for i in range(0,len(loc[0]),1):
        similar_to_loc.append(res[ loc[0][i] ][ loc[1][i] ])    

    # 類似度とx座標とy座標の全てを格納した新たな配列を定義する
    s_x_y = []
    for i in range(0,len(loc[0]),1):
        s = similar_to_loc[i]
        x = loc[1][i]
        y = loc[0][i]
        all_data = [s,x,y]
        s_x_y.append(all_data)    
    s_x_y = sorted(s_x_y,reverse=True)    

    # 次の段階で使用する各種変数の確認
    print("ピーク位置(new_maxid_2[0]):",new_maxid_2[0])
    print("ht,wt = ",ht,wt)

    #new_maxid_2[]
    
    # 各段の中から類似度の高いボタンを4つ選び出す        
    if len(new_maxid_2[0]) == 1:
        P1 = get_button_line_points1(s_x_y,new_maxid_2,ht,wt)
        print("ボタン列が1つ検出されました")
        b_all = [P1]
        #print("P1 = ",P1)
    elif len(new_maxid_2[0]) == 2:
        P1,P2 = get_button_line_points2(s_x_y,new_maxid_2,ht,wt)
        print("ボタン列が2つ検出されました")
        b_all = [P1,P2]
        #print("P1,P2 = ",P1,P2)
    elif len(new_maxid_2[0]) == 3:
        P1,P2,P3 = get_button_line_points3(s_x_y,new_maxid_2,ht,wt)
        print("ボタン列が3つ検出されました")
        #print("P1,P2,P3 = ",P1,P2,P3)
        b_all = [P1,P2,P3]
        
    # ====== ボタン列上のテンプレートマッチングの類似度グラフを描く ====== #
    # ====== 方程式より,直線の座標を格納(この座標を使って, 後ほどテンプレート左上の座標となる)
    # 検出ボタンの4つの中心座標を元に回帰直線を求め, その回帰直線上の類似度マップを計算する    
    # アルゴリズムを記述していく
    count = 1
    for i in b_all:        
        break
        x_points,y_points = get_xy_points(i, w_move, h_move)
        print("1次関数m,n = ",np.polyfit(x_points, y_points, 1)) # numpy 配列でないといけない
        func = np.poly1d(np.polyfit(x_points, y_points, 1))  # 回帰直線の関数を取得
        new_line_crd = get_new_line_crd(func,w_input,w_move,h_move)  # テンプレマッチング用の中心座標をこの配列で定義
        ruijido_list = get_ruijido_list(new_line_crd,FileName,w_move,h_move,wt,ht,temp)
        print("len(ruijido_list)",len(ruijido_list))
        draw_graph(ruijido_list,w_move,w_input,FileName,count) # グラフを保存する
        
        count += 1                
    
    # ピーク検出を行う =============================
    print()
    data_3841_numpy = np.array(md.new_data_3841)
    print("data_3841 の長さ:",len(md.new_data_3841))    
    
    peak = wt    
    maxid = signal.argrelmax(data_3841_numpy, order=peak)
    print("ピーク位置:",maxid[0])

    crd_similar = []
    for i in maxid[0]:
        a = [i,md.data_3841[i]]
        crd_similar.append(a)

    # 検出されたピークの中から域値以上のものだけを取り出す. まずは0.8以上のみを取り出すプログラムを書く
    # (中心座標(検出位置 + w_move), 類似度) の形にする
    print("before crd_similar",crd_similar)        

    new_crd_similar = []
    for i in crd_similar:
        if i[1] >= 0.8:
            new_crd_similar.append(i)
    
    print("after new_crd_similar",new_crd_similar)
    exit()

    
    Fig = plt.figure()
    Map1 = Fig.add_subplot(111)    
    Map1.plot(left, data_3841_numpy)
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')
        
    plt.savefig("./output/{}_jpg/{}_midstep_peak.png".format(FileName, FileName))
    #plt.show()
    
    
    
    
    

    """
    # 自己相関を求める. 相関を求めた後のデータも取得する
    acorr_data = smg.tsa.stattools.acf(height, nlags=len(height), fft=False)
    #plt.stem(np.arange(len(height)), acorr_data)
    #plt.savefig("./output/acorr_data/acf().png")

    print(acorr_data)
    """

    """
    # 自己相関グラフのピークを検出する
    peak = wt
    #maxid = signal.argrelmax(acorr_data, order=peak)
    maxid = signal.argrelmax(acorr_data, order=peak)

    print(maxid[0])
    
    
    Fig = plt.figure()
    Map1 = Fig.add_subplot(111)    
    Map1.plot(left, acorr_data)
    
    Map1.scatter(left[maxid[0]], acorr_data[maxid[0]])
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], acorr_data[maxid[0][i]], 'PEAK!')

    #plt.show()
    #plt.savefig("./output/acorr_data/peak_detect_{}".format(peak))
    """

    
    """
    # テンプレートの2点間の直線のヒストグラムを求める
    t1 = np.array([0,50])
    t2 = np.array([195,50])
    itbuffer = createLineIterator(t1, t2, temp)
    left = itbuffer[0:len(itbuffer), 0]
    height = itbuffer[0:len(itbuffer),2]
    """

    """
    #移動平均を求める
    radius = 50
    height_list = height.tolist()
    height = get_mave(height_list,radius)
    height = np.array(height)

    # 移動平均のウィンドウサイズに合わせて, x軸の個数を整形し直す
    left = np.arange(radius,4031-radius,1)
    """

    """
    # rgb画像にmaxの類似値を持つ結果を出力する
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(res)
    print("類似度最大値:",max_value)
    pt = max_pt
    print("類似度最大の検出座標(左上):",pt)

    # 類似度が最大の出力結果をクロップする
    similar_max_img = img_input[ pt[1] : pt[1]+ht, pt[0] : pt[0]+wt] # img[top : bottom, left : right]
    cv2.imwrite("./output/{}_jpg/max_crop.png".format(FileName), similar_max_img)

    # カラーヒストグラムの比較を行う
    ret = color_hist_compare(temp, similar_max_img)
    print("カラーヒストグラム類似度:{}".format(ret))
    """

    
#=====================================#    




if __name__ == "__main__":
    main()
