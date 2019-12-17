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
# 3511の類似度データを他ファイルから読み込めるようになりました. マッチングの計算を行わなくてもよくなりました
# ピーク検出を行なった類似度グラフを表示できるようになりました

# ver11に追記
# 4245.jpg の画像に対してこのプログラムを適用しました

# ver12に追記
# 3916.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.
# 3841.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.

# ver13に追記
# テンプレートマッチングで検出した個数をグラフにして表示できるようになりました.
# グラフのピーク値から, どこにボタン列があるのかを判定できるようになりました.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as smg
from statsmodels.graphics.tsaplots import plot_acf
import match_data as md # 独自モジュール

# テンプレートのスケールを変換する関数
def get_scale_template(scale,template):    
    scale = scale/10
    # 今は応急処置として,手動でテンプレートのサイズを設定する
    # これは3916.jpgのテンプレサイズです
    w_original = 191
    h_original = 102
    template = cv2.resize(template, dsize=(int(w_original*scale), int(h_original*scale)))
    return template

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


def main(): #=====================================
    FileName = "3841"
    # 入力画像とテンプレート画像をで取得
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    temp = cv2.imread("../../temp_img/IMG_2159.jpg")    

    # グレースケール変換
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # 入力画像の高さ・幅
    h_input,w_input = gray_input.shape
    print(h_input,w_input)

    # テンプレート画像の高さ・幅
    h, w = temp.shape
    print("テンプレの高さ,幅:",h,w)

    # スケール変換実行
    scale = 11
    temp = get_scale_template(scale,temp)    

    ht,wt = temp.shape
        
    # テンプレートマッチング(OpenCV)
    res = cv2.matchTemplate(gray_input,temp,cv2.TM_CCOEFF_NORMED)    
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_input, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)
        
    # rgb画像に域値以上の結果を描画する
    cv2.imwrite("./output/{}_jpg/ikiti_ijou_{}_{}_{}.jpg".format(FileName,scale,FileName,threshold), img_input)

    # 縦軸(検出個数), 横軸(画像の縦サイズ) の類似度グラフを作成する
    left = list(range(0,h_input,1))
    tate_graph = [0]*h_input
    
    h_move = int(ht/2)

    print("検出個数:",len(loc[0]))

    # 左上のy座標ではなく,ボタンの中心座標を格納するよう変更
    new_loc_y = []
    for i in loc[0]:     
        new_loc_y.append(i+h_move)
        tate_graph[i] += 1
        
    plt.bar(left, tate_graph)
    plt.savefig("./output/3841_jpg/num_button_graph/button_num_detect.png")

    # ピークを検出する
    data_3841_numpy = np.array(tate_graph)
    peak = wt    
    maxid = signal.argrelmax(data_3841_numpy, order=peak)
    print("ピーク位置:",maxid[0])
    #for i in maxid[0]:   
    #    print(tate_graph[i])
    
    Fig = plt.figure()
    Map1 = Fig.add_subplot(111)    
    Map1.plot(left, data_3841_numpy)
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')
        
    plt.savefig("./output/{}_jpg/num_button_graph/{}_midstep_peak.png".format(FileName, FileName))
    #plt.show()
    

    

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

    
    # 2点間の直線のヒストグラムを求める # 4245.jpg に合わせている
    P1 = np.array([0,894])
    P2 = np.array([4032,864])
    #itbuffer = createLineIterator(P1, P2, gray_input)
    #left = itbuffer[0:len(itbuffer), 0]
    #height = itbuffer[0:len(itbuffer),2]    

    
    # ====== ボタン列上のテンプレートマッチングの類似度グラフを描く ====== #
    
    # ====== 2点を通る直線の方程式を求める (y = mx + n)
    line = makeLinearEquation(P1[0], P1[1], P2[0], P2[1]) # (x1, y1, x2, y2)
    m = line["m"]
    n = line["n"]

    # ====== 方程式より,直線の座標を格納(この座標を使って, 後ほどテンプレート左上の座標となる)
    line_crd = []
    for x in range(0,w_input,1): #range(0,w_input,1)
        value = (x, int(get_y_crd(m,n,x)))
        line_crd.append(value)
    
    print("スケール変換後のテンプレートの高さ/幅",ht,wt)
    
    # 左上の座標に持って行くための移動幅
    h_move = int(ht/2)
    w_move = int(wt/2)
    print("h_move",h_move)
    print("w_move",w_move)

    # 直線の左端と右端を削除する(テンプレートマッチングとして利用しないため)
    print("line_crd の長さ",len(line_crd))
    new_line_crd = line_crd[w_move:w_input-w_move]
    print("new_line_crd の長さ",len(new_line_crd))
    #print("new_line_crd = {}".format(new_line_crd))

    
    # クロップして,matchテンプレート関数にかける
    ruijido_list = [] # 類似度を格納する配列
    count = 0
    for i in range(len(new_line_crd)):
        break
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

            
    #print(ruijido_list)
    #print(len(ruijido_list))
    
    left = list(range(w_move,w_input-w_move,1))

    #plt.bar(left, ruijido_list)
    #plt.savefig("./output/{}_jpg/{}_ruijido_graph.png".format(FileName,FileName))
    
   
    # ピーク検出を行う
    data_3841_numpy = np.array(md.data_3841)
    print("data_3841 の長さ:",len(md.data_3841))
    peak = wt    
    maxid = signal.argrelmax(data_3841_numpy, order=peak)
    print("ピーク位置:",maxid[0])
    for i in maxid[0]:   
        print(md.data_3841[i])
    
    Fig = plt.figure()
    Map1 = Fig.add_subplot(111)    
    Map1.plot(left, data_3841_numpy)
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')
        
    plt.savefig("./output/{}_jpg/{}_midstep_peak.png".format(FileName, FileName))
    #plt.show()
    """
    
    
    
    

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
    
#=====================================#    




if __name__ == "__main__":
    main()
