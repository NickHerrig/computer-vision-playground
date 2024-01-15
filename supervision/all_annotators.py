import supervision as sv
from inference.models.utils import get_roboflow_model
from tqdm import tqdm


if __name__ == '__main__':

    # Load the yolov8X model from roboflow inference. 
    model = get_roboflow_model('yolov8x-seg-640')

    # Create a byte tracker object to track detections.
    byte_tracker = sv.ByteTrack()

    # get frames iterable from video and loop over them 
    frame_generator = sv.get_video_frames_generator('subway.mp4')

    # get video info from the video path
    video_info = sv.VideoInfo.from_video_path('subway.mp4')
    thickness = sv.calculate_dynamic_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(video_info.resolution_wh)

    # Create the label Annotator, we'll use this on all frames. 
    label = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    # Create a list of the annotators 
    annotators = [
        sv.BoundingBoxAnnotator(thickness=thickness),
        sv.BoxCornerAnnotator(thickness=thickness),
        sv.ColorAnnotator(),
        sv.CircleAnnotator(thickness=thickness),
        sv.DotAnnotator(radius=15),
        sv.TriangleAnnotator(base=25, height=25),
        sv.EllipseAnnotator(thickness=thickness),
        sv.BlurAnnotator(),
        sv.PixelateAnnotator(),
        sv.HeatMapAnnotator(),
        sv.HaloAnnotator(), # Needs sv.Detections.mask
        sv.MaskAnnotator(), # Needs sv.Detections.mask
        sv.PolygonAnnotator(), # Needs sv.Detections.mask
        sv.TraceAnnotator(thickness=thickness), # Needs sv.Detections.tracker_id
    ]

    with sv.VideoSink(target_path="allannotations.mp4", video_info=video_info) as sink:
        frames_per_annotator = video_info.total_frames // len(annotators)
        for i, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):

            # run inference on the frame
            result = model.infer(frame)[0]

            # convert the detections to a supervision detections object. 
            detections = sv.Detections.from_inference(result)
            tracked_detections = byte_tracker.update_with_detections(detections)

            annotator_index = i // frames_per_annotator
            annotator_index = min(annotator_index, len(annotators) - 1)
            
            labels = [ 
                f"{tracker_id} - {model.class_names[class_id]}"
                    for class_id, tracker_id 
                    in zip(tracked_detections.class_id, tracked_detections.tracker_id) 
            ]
            annotated_frame = label.annotate(scene=frame.copy(), detections=tracked_detections, labels=labels)

            if annotator_index == len(annotators) - 1:
                annotated_frame = annotators[annotator_index].annotate(scene=annotated_frame, detections=tracked_detections)
            else:
                annotated_frame = annotators[annotator_index].annotate(scene=annotated_frame, detections=detections)

            sink.write_frame(frame=annotated_frame)