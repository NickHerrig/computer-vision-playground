import supervision as sv
from inference.models.utils import get_roboflow_model
from tqdm import tqdm


if __name__ == '__main__':

    # Load the yolov8X model from roboflow inference. 
    model = get_roboflow_model('yolov8n-640')

    # get video info from the video path and dynamically generate line thickness and text_scale
    video_info = sv.VideoInfo.from_video_path('vehicle.mp4')
    text_scale = sv.calculate_dynamic_text_scale(video_info.resolution_wh)

    # create a ByteTrack object to track detections.
    byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

    # get frames iterable from video and loop over them 
    frame_generator = sv.get_video_frames_generator('vehicle.mp4')

    start_point = sv.Point(0, video_info.height*(3/4))
    end_point = sv.Point(video_info.width, video_info.height*(3/4))

    line_zone = sv.LineZone(start_point, end_point)
    line_annotator = sv.LineZoneAnnotator(color=sv.Color.green(), text_scale=text_scale, custom_in_text="OUT", custom_out_text="IN")

    # create a video sink context manager to write the annotated frames to.
    with sv.VideoSink(target_path="linezone.mp4", video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # run inference on the frame
            result = model.infer(frame)[0]

            # convert the detections to a supervision detections object. 
            detections = sv.Detections.from_inference(result)
            
            # update detections with tracker id's 
            tracked_detections = byte_tracker.update_with_detections(detections)
                        
            #print(tracked_detections)
            line_zone.trigger(detections=tracked_detections)
            annotated_frame = line_annotator.annotate(frame=frame.copy(), line_counter=line_zone)

            # save the annotated frame to the video sink.
            sink.write_frame(frame=annotated_frame)
