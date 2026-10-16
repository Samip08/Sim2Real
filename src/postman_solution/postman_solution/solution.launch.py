from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn(context, *args, **kwargs):
    tier = LaunchConfiguration('tier').perform(context)
    return [Node(package='postman_solution', executable='tier%s_node' % tier,
                 name='tier%s_node' % tier, output='screen',
                 parameters=[{'use_sim_time': True}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('tier', default_value='1'),
        OpaqueFunction(function=_spawn),
    ])
