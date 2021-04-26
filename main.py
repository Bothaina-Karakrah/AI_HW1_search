from framework.graph_search.astar import AStar
from framework import * 
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional
import os


# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map_cur_speeds.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):

    # TODO [Ex.20]: Complete the implementation of this method.

    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    p1, = ax1.plot(weights, total_cost, '-b', label='Solution cost')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    p2, = ax2.plot(weights, total_nr_expanded, '-r',  label='#Expanded states')

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 40_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):
    # TODO [Ex.20]:

    weights = []
    total_cost = []
    total_nr_expanded = []

    weights_arr = np.linspace(low_heuristic_weight, high_heuristic_weight, n)

    for weight in weights_arr:
        a_star = AStar(heuristic_type, weight, max_nr_states_to_expand)
        result = a_star.solve_problem(problem)
        if result.is_solution_found:
            weights.append(weight)
            total_cost.append(result.solution_cost)
            total_nr_expanded.append(result.nr_expanded_states)
    plot_distance_and_expanded_wrt_weight_figure(problem.name, weights, total_cost, total_nr_expanded)


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
    if not hasattr(solver, '__focal_heuristic'):
        setattr(solver, '__focal_heuristic', HistoryBasedHeuristic(problem=problem))
    focal_heuristic = getattr(solver, '__focal_heuristic')
    return focal_heuristic.estimate(node.state)


def toy_map_problem_experiment():
    print()
    print('Solve the distance-based map problem.')

    # TODO [Ex.7]: Just run it and inspect the printed result.

    target_point = 549
    start_point = 82700
    
    dist_map_problem = MapProblem(streets_map, start_point, target_point, 'distance') 
    
    uc = UniformCost()
    res = uc.solve_problem(dist_map_problem)
    print(res)
    
    # save visualization of the path
    file_path = os.path.join(Consts.IMAGES_PATH, 'UCS_path_distance_based.png')
    streets_map.visualize(path=res, file_path=file_path)


def map_problem_experiments():
    print()
    print('Solve the map problem.')

    target_point = 549
    start_point = 82700
    # TODO [Ex.12]: 1. create an instance of `MapProblem` with a current_time-based operator cost
    #           with the start point `start_point` and the target point `target_point`
    #           and name it `map_problem`.
    #       2. create an instance of `UCS`,
    #           solve the `map_problem` with it and print the results.
    #       3. save the visualization of the path in 'images/UCS_path_time_based.png'
    # You can use the code in the function 'toy_map_problem_experiment' for help.

    map_problem = MapProblem(streets_map, start_point, target_point, 'current_time')

    uc = UniformCost()
    res = uc.solve_problem(map_problem)
    print(res)

    # save visualization of the path
    file_path = os.path.join(Consts.IMAGES_PATH, 'UCS_path_time_based.png')
    streets_map.visualize(path=res, file_path=file_path)

    # TODO [Ex.16]: create an instance of `AStar` with the `NullHeuristic`

    a_star = AStar(NullHeuristic)
    res = a_star.solve_problem(map_problem)
    print(res)

    # TODO [Ex.18]: create an instance of `AStar` with the `TimeBasedAirDistHeuristic`,

    a_star = AStar(TimeBasedAirDistHeuristic)
    res = a_star.solve_problem(map_problem)
    print(res)

    # TODO [Ex.20]:

    run_astar_for_weights_in_range(TimeBasedAirDistHeuristic, map_problem)

    # TODO [Ex.24]: 1. Call the function set_additional_shortest_paths_based_data()
    #                   to set the additional shortest-paths-based data in `map_problem`.
    #                   For more info see `problems/map_problem.py`.
    #               2. create an instance of `AStar` with the `ShortestPathsBasedHeuristic`,
    #                  solve the same `map_problem` with it and print the results (as before).

    map_problem.set_additional_shortest_paths_based_data()
    a_star = AStar(ShortestPathsBasedHeuristic)
    res = a_star.solve_problem(map_problem)
    print(res)
    
    # TODO [Ex.25]: 1. Call the function set_additional_history_based_data()
    #                   to set the additional history-based data in `map_problem`.
    #                   For more info see `problems/map_problem.py`.
    #               2. create an instance of `AStar` with the `HistoryBasedHeuristic`,
    #                   solve the same `map_problem` with it and print the results (as before).

    map_problem.set_additional_history_based_data()
    a_star = AStar(HistoryBasedHeuristic)
    res = a_star.solve_problem(map_problem)
    print(res)

    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    # TODO [Ex.29]: Create an instance of `AStarEpsilon` with the `ShortestPathsBasedHeuristic`.
    #       Solve the `map_problem` with it and print the results.
    #       Use focal_epsilon=0.23, and max_focal_size=40.
    #       Use within_focal_priority_function=within_focal_h_sum_priority_function. This function
    #        (defined just above) is internally using the `HistoryBasedHeuristic`.

    a_star_epsilon = AStarEpsilon(ShortestPathsBasedHeuristic, within_focal_h_sum_priority_function, 0.5, None, 0.23, 40)


def run_all_experiments():
    print('Running all experiments')
    toy_map_problem_experiment()
    map_problem_experiments()


if __name__ == '__main__':
    run_all_experiments()
