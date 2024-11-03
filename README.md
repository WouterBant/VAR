# Vision for Autonomous Robots Assignments

In your ros2 workspace:
```bash
git clone https://github.com/WouterBant/VAR.git
```

Setup environment:
```
call install\setup.bat
```

Set debugging mode with:
```
set DEBUG=[0,1,2,3,4,...]
```

Code formatter / linter:
```
ruff format src
```

## Lab 1: Line Following
Build with:
```bash
colcon build --packages-select lab1
```

Run with:
```bash
ros2 run lab1 line_follower
```

# Lab2:
Note: ros2 pkg create --build-type ament_python --node-name my_node lab2 in the src folder