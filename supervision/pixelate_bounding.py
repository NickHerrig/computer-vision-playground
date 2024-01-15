import supervision as sv
from inference.models.utils import get_roboflow_model


if __name__ == '__main__':

    # Load the yolov8X model from roboflow inference. 
    model = get_roboflow_model('yolov8x-640')

    # get video info from the video path and dynamically generate line thickness
    video_info = sv.VideoInfo.from_video_path('vehicle.mp4')
    thickness = sv.calculate_dynamic_line_thickness(video_info.resolution_wh)

    # create a bounding box annotator with dynamic thickness and a pixelate annotator.
    bounding_box = sv.BoundingBoxAnnotator(thickness=thickness)
    pixalate = sv.PixelateAnnotator()

    # get frames iterable from video and loop over them 
    frame_generator = sv.get_video_frames_generator('vehicle.mp4')

    # create a video sink context manager to write the annotated frames to.
    with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
        for frame in frame_generator:

            # run inference on the frame
            result = model.infer(frame)[0]

            # convert the detections to a supervision detections object. 
            detections = sv.Detections.from_inference(result)

            #  apply pixalate on frame copy, then add bounding box. 
            annotated_frame = pixalate.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = bounding_box.annotate(scene=annotated_frame, detections=detections)

            # save the annotated frame to the video sink.
            sink.write_frame(frame=annotated_frame)