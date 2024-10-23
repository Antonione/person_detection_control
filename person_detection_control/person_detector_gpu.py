import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ur3e_interface_msgs.msg import PeopleInfo, PersonInfo  # Importa as mensagens customizadas
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge


class PersonDetectionNode(Node):
    def __init__(self):
        super().__init__('person_detection_node')

        # Subscrição dos tópicos de imagem e profundidade
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Publisher para enviar as distâncias e bounding boxes das pessoas
        self.people_info_pub = self.create_publisher(PeopleInfo, 'people_info', 10)

        # Inicializando o CvBridge para converter mensagens ROS <-> OpenCV
        self.bridge = CvBridge()

        # Inicializar variáveis de frames de imagem e profundidade
        self.color_image = None
        self.depth_image = None

        # Configuração do modelo Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Limite de confiança
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        self.get_logger().info(f'Device usado: {self.cfg.MODEL.DEVICE}')
        self.get_logger().info('Nó de detecção de pessoas iniciado.')

    def color_callback(self, msg):
        # Converte a mensagem ROS de imagem colorida para OpenCV
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Processa a imagem se ambos os frames (cor e profundidade) estiverem disponíveis
        if self.depth_image is not None:
            self.process_frame()

    def depth_callback(self, msg):
        # Converte a mensagem ROS de imagem de profundidade para OpenCV e transforma em tensor
        depth_image_cv = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_image = torch.from_numpy(depth_image_cv).float().cuda()  # Mantém na GPU

        # Processa a imagem se ambos os frames (cor e profundidade) estiverem disponíveis
        if self.color_image is not None:
            self.process_frame()

    def process_frame(self):
        outputs = self.predictor(self.color_image)
        instances = outputs["instances"]
        person_indices = instances.pred_classes == 0  # Classe "0" corresponde a pessoas
        persons = instances[person_indices]

        people_info_msg = PeopleInfo()
        window_size = 20

        if len(persons) > 0:
            # Mantém os dados na GPU
            masks = persons.pred_masks  # Tensor PyTorch
            boxes = persons.pred_boxes.tensor  # Tensor PyTorch

            for i, mask in enumerate(masks):
                coords = torch.nonzero(mask)  # Usa torch para obter coordenadas

                # Central da máscara usando operações PyTorch
                center_y, center_x = torch.mean(coords[:, 0].float()), torch.mean(coords[:, 1].float())
                y_min, y_max = int(center_y - window_size), int(center_y + window_size)
                x_min, x_max = int(center_x - window_size), int(center_x + window_size)

                # Extrai valores de profundidade diretamente em tensores PyTorch
                depth_values_central = self.depth_image[y_min:y_max, x_min:x_max]
                valid_depths_central = depth_values_central[depth_values_central > 0]
                distance_central_region = torch.mean(valid_depths_central) / 1000 if valid_depths_central.numel() > 0 else None

                # Adiciona a pessoa à lista de PersonInfo
                person_info = PersonInfo()
                person_info.id = i
                person_info.distance_central_region = distance_central_region.item() if distance_central_region else -1.0
                person_info.x_min = int(boxes[i][0])
                person_info.y_min = int(boxes[i][1])
                person_info.x_max = int(boxes[i][2])
                person_info.y_max = int(boxes[i][3])

                # Adiciona à lista de pessoas
                people_info_msg.people.append(person_info)

        # Publica a mensagem de PeopleInfo contendo todas as pessoas detectadas
        self.people_info_pub.publish(people_info_msg)

        # Visualizar os resultados
        self.visualize_results(persons, people_info_msg)

    def visualize_results(self, persons, people_info_msg):
        # Certifique-se de que a janela seja criada apenas uma vez
        if not hasattr(self, 'window_created'):
            cv2.namedWindow("Segmentação com Distância", cv2.WINDOW_NORMAL)
            self.window_created = True

        # Visualizar resultados com PyTorch
        v = Visualizer(self.color_image[:, :, ::-1], metadata=self.metadata, scale=1.2)
        out = v.draw_instance_predictions(persons.to("cpu"))  # Move para CPU apenas para visualização com OpenCV
        result_frame = out.get_image()[:, :, ::-1].copy()

        for person_info in people_info_msg.people:
            x_min, y_min, x_max, y_max = person_info.x_min, person_info.y_min, person_info.x_max, person_info.y_max
            offset_y = 30 * person_info.id
            if person_info.distance_central_region >= 0:
                cv2.putText(result_frame, f'Pessoa {person_info.id + 1} - Central: {person_info.distance_central_region:.2f} m',
                            (10, 30 + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Atualiza o frame na janela existente
        cv2.imshow("Segmentação com Distância", result_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
