import supervision as sv
from inference.models.utils import get_roboflow_model
import numpy as np
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

    polygon = np.array([[9, 1758],[1125, 846],[1697, 850],[1885, 2146],[17, 2146],[17, 1754]])

    polygon_zone = sv.PolygonZone(polygon, frame_resolution_wh=video_info.resolution_wh)
    polygon_annotator = sv.PolygonZoneAnnotator(color=sv.Color.green(), zone=polygon_zone, text_scale=text_scale)

    box_annotator = sv.BoxAnnotator(text_scale=text_scale)

    with sv.ImageSink(target_dir_path=".") as sink:
        frame = next(frame_generator)
        sink.save_image(frame)

    # create a video sink context manager to write the annotated frames to.
    with sv.VideoSink(target_path="polygonzone.mp4", video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # run inference on the frame
            result = model.infer(frame)[0]

            # convert the detections to a supervision detections object. 
            detections = sv.Detections.from_inference(result)

            # filter based on the polygon zone. 
            detections = detections[polygon_zone.trigger(detections)]
            
            # update detections with tracker id's 
            tracked_detections = byte_tracker.update_with_detections(detections)

            polygon_zone.trigger(tracked_detections)
            annotated_frame = polygon_annotator.annotate(scene=frame.copy())
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)

            # save the annotated frame to the video sink.
            sink.write_frame(frame=annotated_frame)
