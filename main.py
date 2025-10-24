import argparse
from pathlib import Path
from tracking import process_video


def main():
    """
    Главная точка входа в программу.
    Парсит аргументы командной строки и запускает обработку видео.
    """
    parser = argparse.ArgumentParser(
        description="Детекция, трекинг и LERP-интерполяция людей на видео."
    )
    
    parser.add_argument(
        '--input', type=str, default='input/crowd.mp4',
        help="Путь к исходному видеофайлу. [cite: 3]"
    )
    parser.add_argument(
        '--output', type=str, default='output/crowd.mp4',
        help="Путь для сохранения результата. [cite: 13]"
    )
    parser.add_argument(
        '--model', type=str, default='yolov8s.pt',
        help="Имя или путь к модели YOLO (например, yolov8n.pt, yolov8s.pt)."
    )
    parser.add_argument(
        '--conf_thresh', type=float, default=0.3,
        help="Порог уверенности (confidence) для детекции."
    )
    parser.add_argument(
        '--frame_skip', type=int, default=10,
        help="Обрабатывать каждый N-й кадр."
    )
    parser.add_argument(
        '--iou_thresh', type=float, default=0.3,
        help="Порог IoU для матчинга."
    )
    parser.add_argument(
        '--max_age', type=int, default=50,
        help="Макс. кол-во кадров, которое трек живет без детекции."
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Input video path: {input_path}")
    print(f"Output video path: {output_path}")

    process_video(
        video_path=input_path, 
        output_path=output_path, 
        model_name=args.model,
        conf_threshold=args.conf_thresh,
        frame_skip=args.frame_skip,
        iou_threshold=args.iou_thresh,
        max_age_frames=args.max_age
    )

if __name__ == "__main__":
    main()