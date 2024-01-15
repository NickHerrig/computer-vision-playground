import supervision as sv
from inference.models.utils import get_roboflow_model


if __name__ == '__main__':

    # Load the yolov8X model from roboflow inference. 
    model = get_roboflow_model('yolov8x-seg-640')

    # get video info from the video path and dynamically generate line thickness and text_scale
    video_info = sv.VideoInfo.from_video_path('vehicle.mp4')
    thickness = sv.calculate_dynamic_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(video_info.resolution_wh)

    # create a trace and label annotator, with dynamic video info.
    trace = sv.TraceAnnotator(thickness=thickness)
    label = sv.LabelAnnotator(text_thickness=thickness, text_scale=text_scale)
    polygon = sv.PolygonAnnotator(thickness=thickness)

    # create a ByteTrack object to track detections.
    byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

    # get frames iterable from video and loop over them 
    frame_generator = sv.get_video_frames_generator('vehicle.mp4')

    # create a video sink context manager to write the annotated frames to.
    with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
        for frame in frame_generator:

            # run inference on the frame
            result = model.infer(frame)[0]

            # convert the detections to a supervision detections object. 
            detections = sv.Detections.from_inference(result)

            annotated_frame = polygon.annotate(scene=frame.copy(), detections=detections)
            
            # update detections with tracker id's 
            tracked_detections = byte_tracker.update_with_detections(detections)

            #  apply trace annotator to  frame.
            annotated_frame = trace.annotate(scene=annotated_frame, detections=tracked_detections)

            # create label text for annotator
            labels = [ f"{tracker_id}" for tracker_id in tracked_detections.tracker_id ]

            # apply label annotator to frame
            annotated_frame = label.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

            # save the annotated frame to the video sink.
            sink.write_frame(frame=annotated_frame)