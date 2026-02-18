import os
import csv
from pathlib import Path

from main import CONFIG, Car, GeneticPopulation, DQNAgent, device  # reuse core logic
from track_setup import create_track


# Configuration for the N-sides batch experiment
N_SIDES_LIST = [ 4, 5, 6]  # Triangle, square, pentagon, hexagon (edit as needed)
GA_MAX_GENERATIONS = 40
DQN_MAX_EPISODES = 300
TARGET_CIRCLES = 1  # "Success" = at least this many full laps in one run
OUTPUT_PATH = Path("bench") / "n_sides_comparison.csv"


def ensure_bench_dir():
    bench_dir = Path("bench")
    bench_dir.mkdir(exist_ok=True)
    return bench_dir


def run_ga_for_n_sides(n_sides, max_generations=GA_MAX_GENERATIONS, target_circles=TARGET_CIRCLES):
    """
    Train GA on a track with n_sides and return:
      - first generation index where any agent achieves >= target_circles
        (None if never reached within max_generations)
    """
    track = create_track(CONFIG["TRACK_SIZE"], CONFIG["TRACK_WIDTH"], n_sides)
    # Use a dedicated GA log file per N so logs don't overwrite each other
    ga_log_path = f"bench/ga_log_n{n_sides}.txt"
    ga = GeneticPopulation(log_path=ga_log_path)

    first_success_generation = None

    for _ in range(max_generations):
        scored_pop = ga.evaluate(track)
        best_circles = max(s[1] for s in scored_pop)
        print(f"GA (N={n_sides}) Gen {ga.gen_count} | best circles: {best_circles}")

        if best_circles >= target_circles:
            # Record the generation where we first hit the target and stop early
            first_success_generation = ga.gen_count
            break

        ga.evolve(scored_pop)

    return first_success_generation


def run_dqn_for_n_sides(n_sides, max_episodes=DQN_MAX_EPISODES, target_circles=TARGET_CIRCLES):
    """
    Train DQN on a track with n_sides and return:
      - first episode index where the agent achieves >= target_circles
        (None if never reached within max_episodes)
    """
    track = create_track(CONFIG["TRACK_SIZE"], CONFIG["TRACK_WIDTH"], n_sides)
    # Use a dedicated DQN log file per N so logs don't overwrite each other
    dqn_log_path = f"bench/dqn_log_n{n_sides}.txt"
    agent = DQNAgent(log_path=dqn_log_path)
    FRAME_SKIP = 4

    first_success_episode = None

    while agent.episode < max_episodes:
        car = Car(track)
        state_tensor = car.get_state()
        state = state_tensor.cpu().numpy() if hasattr(state_tensor, "cpu") else state_tensor
        total_reward = 0

        while car.alive and car.time_alive < 10000:
            action = agent.select_action(
                state_tensor if isinstance(state_tensor, type(state_tensor)) else
                (state_tensor if hasattr(state_tensor, "to") else state)
            )

            reward_accum = 0
            for _ in range(FRAME_SKIP):
                next_state_tensor, r, done, _ = car.step(action)
                next_state = (
                    next_state_tensor.cpu().numpy()
                    if hasattr(next_state_tensor, "cpu")
                    else next_state_tensor
                )
                reward_accum += r
                if done:
                    break

            agent.memory.push(state, action, reward_accum, next_state, done)
            state = next_state
            state_tensor = next_state_tensor
            total_reward += reward_accum

            agent.optimize_model()

        # Episode finished
        if car.circles >= target_circles and first_success_episode is None:
            first_success_episode = agent.episode
            print(f"DQN (N={n_sides}) reached {target_circles} circles at episode {first_success_episode}")
            agent.log_episode(total_reward, car.time_alive, car.circles)
            break

        agent.log_episode(total_reward, car.time_alive, car.circles)

    return first_success_episode


def run_experiments(
    n_sides_list=N_SIDES_LIST,
    ga_max_generations=GA_MAX_GENERATIONS,
    dqn_max_episodes=DQN_MAX_EPISODES,
    target_circles=TARGET_CIRCLES,
    output_path=OUTPUT_PATH,
):
    ensure_bench_dir()

    rows = []

    for n in n_sides_list:
        print(f"\n=== Running experiments for N={n} sides ===")

        # GA
        ga_first_gen = run_ga_for_n_sides(
            n_sides=n,
            max_generations=ga_max_generations,
            target_circles=target_circles,
        )
        if ga_first_gen is None:
            print(f"GA did not reach {target_circles} circles within {ga_max_generations} generations for N={n}.")
        else:
            print(f"GA first reached {target_circles} circles at generation {ga_first_gen} for N={n}.")

        # Convert generations to "episode equivalents" using the same 50× heuristic
        ga_episodes_equiv = ga_first_gen * 50 if ga_first_gen is not None else None

        # DQN
        dqn_first_ep = run_dqn_for_n_sides(
            n_sides=n,
            max_episodes=dqn_max_episodes,
            target_circles=target_circles,
        )
        if dqn_first_ep is None:
            print(f"DQN did not reach {target_circles} circles within {dqn_max_episodes} episodes for N={n}.")
        else:
            print(f"DQN first reached {target_circles} circles at episode {dqn_first_ep} for N={n}.")

        rows.append(
            {
                "n_sides": n,
                "ga_first_generation_to_target": ga_first_gen if ga_first_gen is not None else -1,
                "ga_episodes_equiv_to_target": ga_episodes_equiv if ga_episodes_equiv is not None else -1,
                "dqn_episodes_to_target": dqn_first_ep if dqn_first_ep is not None else -1,
                "target_circles": target_circles,
                "ga_max_generations": ga_max_generations,
                "dqn_max_episodes": dqn_max_episodes,
            }
        )

    # Write CSV summary
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_sides",
                "ga_first_generation_to_target",
                "ga_episodes_equiv_to_target",
                "dqn_episodes_to_target",
                "target_circles",
                "ga_max_generations",
                "dqn_max_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved N-sides comparison summary to {output_path}")


if __name__ == "__main__":
    run_experiments()

