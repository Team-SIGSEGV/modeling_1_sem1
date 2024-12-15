import numpy as np
import matplotlib.pyplot as plt

import configparser

config = configparser.ConfigParser()

config.read('params.ini', encoding='utf-8')

def get_value_from_PARAMS(value):
    return config.getfloat("PARAMS", value)

def get_QuantityScenario():
    return config.getint("QuantityScenario", "n")

def get_scenario_from_ini(value):
     k = config.getfloat(f'scenario{value}', 'k')
     engine_on = config.getboolean(f'scenario{value}', 'engine_on')
     thrust = config.getfloat(f'scenario{value}', 'thrust')
     initial_state = list(map(float, config.get(f'scenario{value}', 'initial_state').split(',')))
     title = config.get(f'scenario{value}', 'title')

     return k, engine_on, thrust, initial_state, title


# Константы (те же, что и раньше)
G = get_value_from_PARAMS("G")
M = get_value_from_PARAMS("M")
m = get_value_from_PARAMS("m_t")
dt = get_value_from_PARAMS("dt")  # Шаг по времени
T = get_value_from_PARAMS("T")  # Общее время моделирования (может потребоваться корректировка)

# Функция f (без изменений)
def f(t, state, k, engine_on, thrust):
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    ax = -G * M * x / r ** 3 - k * vx / m
    ay = -G * M * y / r ** 3 - k * vy / m
    if engine_on:
        angle = np.arctan2(vy, vx)  # Направление тяги совпадает с направлением скорости
        ax += thrust * np.cos(angle) / m
        ay += thrust * np.sin(angle) / m
    return [vx, vy, ax, ay]


# ... (та же функция, что и в предыдущем ответе)

# Метод Рунге-Кутты (без изменений)
def rk4(t, state, k, engine_on, thrust, dt):
    k1 = np.array(f(t, state, k, engine_on, thrust))
    k2 = np.array(f(t + dt / 2, state + dt / 2 * k1, k, engine_on, thrust))
    k3 = np.array(f(t + dt / 2, state + dt / 2 * k2, k, engine_on, thrust))
    k4 = np.array(f(t + dt, state + dt * k3, k, engine_on, thrust))
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ... (та же функция, что и в предыдущем ответе)

# --- Моделирование и визуализация ---
fig, ax = plt.subplots(figsize=(12, 8))


def simulate(k, initial_state, engine_on_check, thrust):
    t_points = np.arange(0, T, dt)
    states = np.zeros((len(t_points), 4))
    states[0] = initial_state
    engine_on = False  # Изначально двигатель выключен

    for i in range(len(t_points) - 1):
        states[i + 1] = rk4(t_points[i], states[i], k, engine_on, thrust, dt)
        engine_on = engine_on_check  # Проверка условия включения двигателя
    return states[:, 0], states[:, 1]


for i in range(0, get_QuantityScenario()):
    k, engine_on, thrust, initial_state, title = get_scenario_from_ini(i + 1)
    x, y = simulate(k, initial_state, engine_on, thrust)

    ax.plot(x, y)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    plt.savefig(f'out/image{i+1}.png')
    plt.cla()