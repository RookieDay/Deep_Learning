import base64
import struct
import os

def readline(line):
    MID,ImageSearchRank,ImageURL,PageURL,FaceID,FaceRectangle,FaceData=line.split("\t")
    rect=struct.unpack("ffff",base64.b64decode(FaceRectangle))
    return MID,ImageSearchRank,ImageURL,PageURL,FaceID,rect,base64.b64decode(FaceData)

def writeImage(filename,data):
    with open(filename,"wb") as f:
        f.write(data)

def unpack(filename,target="img"):
    i=0
    with open(filename,"r",encoding="utf-8") as f:
        for line in f:
            MID,ImageSearchRank,ImageURL,PageURL,FaceID,FaceRectangle,FaceData=readline(line)
            img_dir=os.path.join(target,MID)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            img_name="%s-%s"%(ImageSearchRank,FaceID)+".jpg"
            writeImage(os.path.join(img_dir,img_name),FaceData)
            i+=1
            if i%1000==0:
                print(i,"imgs finished")
    print("all finished")    


filename="MsCelebV1-Faces-Aligned.tsv" 
unpack(filename)
