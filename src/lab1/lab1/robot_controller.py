import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
import time
import threading

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create a publisher for cmd_vel topic
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Flag to control movement
        self.is_moving = False
        self.movement_thread = None

    def move_forward(self, speed=0.5):
        """Move robot forward continuously"""
        self.is_moving = True
        self.movement_thread = threading.Thread(target=self._continuous_move, 
                                                args=(speed, 'forward'))
        self.movement_thread.start()

    def move_backward(self, speed=0.5):
        """Move robot backward continuously"""
        self.is_moving = True
        self.movement_thread = threading.Thread(target=self._continuous_move, 
                                                args=(speed, 'backward'))
        self.movement_thread.start()

    def turn_left(self, angular_speed=0.5):
        """Rotate robot left continuously"""
        self.is_moving = True
        self.movement_thread = threading.Thread(target=self._continuous_move, 
                                                args=(angular_speed, 'left'))
        self.movement_thread.start()

    def turn_right(self, angular_speed=0.5):
        """Rotate robot right continuously"""
        self.is_moving = True
        self.movement_thread = threading.Thread(target=self._continuous_move, 
                                                args=(angular_speed, 'right'))
        self.movement_thread.start()

    def _continuous_move(self, speed, direction):
        """Continuously publish movement commands"""
        while self.is_moving:
            cmd = Twist()
            
            if direction == 'forward':
                cmd.linear = Vector3(x=float(speed))
            elif direction == 'backward':
                cmd.linear = Vector3(x=float(-speed))
            elif direction == 'left':
                cmd.angular = Vector3(z=float(speed))
            elif direction == 'right':
                cmd.angular = Vector3(z=float(-speed))
            
            self.publisher_.publish(cmd)
            time.sleep(0.1)  # Publish at 10 Hz

    def stop(self):
        """Stop the robot"""
        self.is_moving = False
        
        # Wait for movement thread to finish
        if self.movement_thread:
            self.movement_thread.join()
        
        # Send stop command
        cmd = Twist()
        cmd.linear = Vector3(x=0.0)
        cmd.angular = Vector3(z=0.0)
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    
    try:
        print("Robot Controller")
        print("Commands:")
        print("w - Move Forward")
        print("s - Move Backward")
        print("a - Turn Left")
        print("d - Turn Right")
        print("q - Stop")
        print("x - Exit")
        
        while True:
            command = input("Enter command: ").lower()
            
            if command == 'w':
                robot_controller.move_forward()
            elif command == 's':
                robot_controller.move_backward()
            elif command == 'a':
                robot_controller.turn_left(angular_speed=3.5)
            elif command == 'd':
                robot_controller.turn_right()
            elif command == 'q':
                robot_controller.stop()
            elif command == 'x':
                robot_controller.stop()
                break
            else:
                print("Invalid command")
    
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Stopped by user')
    
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()