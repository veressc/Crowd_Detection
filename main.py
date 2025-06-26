from detect import detect_people_in_video

if __name__ == "__main__":
    input_video = "crowd.mp4"
    output_video = "output.mp4"
    detect_people_in_video(input_video, output_video)
