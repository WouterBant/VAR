# Vision for Autonomous Robots Assignments

In your ros2 workspace:
```bash
git clone https://github.com/WouterBant/VAR.git
```

Setup environment:
```
call install\setup.bat
```

Code formatter / linter:
```
ruff format && ruff check
```

## Lab 1: Line Following
[Config file](configs/lab1/config.yaml)

Build with:
```bash
colcon build --packages-select lab1
```

Run with:
```bash
ros2 run lab1 line_follower
```

## Lab2: Detection and Localization
[Config file](configs/lab2/config.yaml)

Build with:
```bash
colcon build --packages-select lab2
```

Run with:
```bash
ros2 run lab2 curling
```


## Lab 3:
[Config file](configs/lab3/config.yaml)

Build with:
```bash
colcon build --packages-select lab3
```

Run with:
```bash
ros2 run lab3 my_node
```