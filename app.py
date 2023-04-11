from face_detector import YoloDetector
import cv2
import cvzone
import streamlit as st
import time
tab0 , tab1 = st.tabs(["Home" , "Detection"])
with st.sidebar:
    st.image("icon.png" ,width=100)
    detect_from = st.selectbox("Detect Faces from : " ,
                                ["File" , "Live"])
    device = st.selectbox("CPU / GPU : " , ["cpu" , "gpu"])
    if device == "cpu" : 
        device_name = "cpu"
    elif device == "gpu" : 
        index = st.selectbox("Your cuda Index : " , (1,2,3))
        device_name = f"cuda:{index}" 
    save = st.radio("Do you want to save results ? " , 
                    ("Yes" , "No"))
    iou = st.slider("Select threshold for NMS : " , value=0.5 , min_value=0.1 , max_value=1.0)
    conf_thres = st.slider("Select The confidence threshold : " , value=0.3 , min_value=0.1 , max_value=1.0)
with tab0:
    st.header("About This Project : ")
    st.image("home.jpg")
    st.write("""Face Blur using YOLOFace is a cutting-edge computer vision system that provides an automated and efficient solution
        for blurring faces in images or videos. Leveraging the advanced YOLOFace algorithm, it can accurately detect and localize faces 
        in real-time, allowing for precise and reliable face blurring. With its high accuracy and speed, Face Blur using YOLOFace can 
        be integrated into various applications, such as privacy protection in surveillance footage, media censorship, or data 
        anonymization in compliance with privacy regulations. This system offers a user-friendly interface and flexible configuration
        options, making it customizable for different use cases. It ensures the protection of individuals' identities while preserving
        the integrity of the visual content. The YOLOFace-based approach ensures real-time performance without compromising on 
        accuracy, making it a state-of-the-art solution for face blurring needs. The system is designed to be easily integrated 
        into existing workflows, making it a powerful tool for privacy-conscious organizations or individuals who need to anonymize
        faces in images or videos quickly and effectively. With Face Blur using YOLOFace, you can confidently protect privacy while
        maintaining visual quality in your multimedia content.
""")
with tab1 : 
    if detect_from == "File" : 
        source = st.file_uploader("Upload Ur Video : " ,
                                   type=["mp4" , "mvi"])
        if source : 
            source = source.name
    else : 
        LIVE = st.selectbox("Select Live type : " , 
                            ["WebCam" , "URL"])
        if LIVE == "WebCam" :
            source = st.selectbox("Select Your Index Device : " , 
                                  (1,2,3))

        else : 
            source = st.text_input("Entre Your Url here :")

    col1 , _ , col2  = st.columns(3)

    with col1 : 
        bstart = st.button("Start")
    with col2 : 
        if save == "Yes" :     
            bstop = st.button("Save")
        elif save == "No" : 
            bstop = st.button("Stop")


    if bstart :
        model = YoloDetector(target_size=720, device=device_name, min_face=100)
        cap = cv2.VideoCapture(source)
        if bstart : 
            cap = cv2.VideoCapture(source)
            if save == "Yes" : 
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
                out = cv2.VideoWriter(f'results/{str(time.asctime())}.mp4',
                                        fourcc, 10, (w, h))
        frame_window = st.image( [] )

        while True : 
            _ , img = cap.read()
            results,points = model.predict(img , iou_thres=iou ,conf_thres=conf_thres )
            for result in results : 
                for bbox in result : 
                    h = (bbox[2] - bbox[0])
                    w = (bbox[3] - bbox[1])
                    crop = img[bbox[1]:bbox[1]+h , bbox[0]:bbox[0]+w] 
                    cvzone.cornerRect(img ,(bbox[0] , bbox[1],w,h ) ,l=10)
                    blur = cv2.blur(crop , (50,50))
                    img[bbox[1]:bbox[1]+h , bbox[0]:bbox[0]+w] = blur
                try : 
                    out.write(img)
                except : 
                    pass
                img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
                frame_window.image(img)
            if bstop : 
                try : 
                    cap.release()
                except : 
                    pass