from flask import Flask, render_template, Response, request
from game import *
from spshands import *
# from camera1 import *
import datetime
import cv2
import json

import os

hand=0
global capture,end,botplayed,youplayed,resultp,resultb,res,displaywin,bscore,pscore
capture=0
end=0
displaywin=""
botplayed=[]
youplayed=[]
resultp=[]
resultb=[]
res=[]
pscore=0
bscore=0

app = Flask(__name__)
camera = cv2.VideoCapture(0)



def updateScore(play,bplay,p,b):
    winRule = {'Rock':'Scissors','Scissors':'Paper','Paper':'Rock'}
    if play == bplay:
        return p,b
    elif bplay == winRule[play]:
        return p+1,b
    else:
        return p,b+1

def dispwinner(playerScore,botScore):
	if playerScore > botScore:
		winner = "You Won!!"
	elif playerScore == botScore:
		winner = "Its a Tie"
	else:
		winner = "Bot Won.."
	return winner
		

def gen_frames():  # generate frame by frame from camera
	global capture,pscore,bscore,end,camera,botplayed,youplayed,resultp,resultb,res,displaywin
	
	while True:
		success, frame = camera.read()
		if success:   
			if(capture):
				capture=0
				# p = "shots/shot.jpg"
				detector = handDetector()
				frame = detector.findHands(frame)
				boxcord=detector.findPosition(frame)
				frame=frame[boxcord[1]:boxcord[3],boxcord[0]:boxcord[2]]
				# cv2.imwrite(p, frame)
				pred,bplay=play(frame)
				botplayed.append(bplay)
				youplayed.append(pred)
				pscore,bscore=updateScore(pred,bplay,pscore,bscore)
				resultp.append(pscore)
				resultb.append(bscore)
				displaywin=dispwinner(pscore,bscore)
				# print(pscore,bscore,bplay)

			
			ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
			frame = buffer.tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')


@app.route('/',methods=['GET','POST'])
def index(): 
	pscore=0
	bscore=0     
	return render_template("index.html")

@app.route('/playgame')
def playgame():
    return render_template("playgame.html")

@app.route('/video_feed',methods=['GET','POST'])
def video_feed():
	return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
	global switch,camera
	if request.method == 'POST':
		if request.form.get('click') == 'capture':
			global capture
			capture=1
		if request.form.get('click') == 'end':
			global end,res,displaywin
			end=1

	elif request.method=='GET':
		return render_template('playgame.html')
	return render_template('playgame.html')

@app.route('/result',methods=['GET','POST'])
def result():      
	# print("final",pscore,bscore,displaywin)
	return render_template("result.html",res1=pscore,res2=bscore,res3=displaywin,res4=botplayed,res5=youplayed,res6=resultb,res7=resultp)
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)

camera.release()
cv2.destroyAllWindows()     