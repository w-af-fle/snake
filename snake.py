import numpy as np
import pygame
import random
import sys
from collections import deque

# ================== 1. НАСТРОЙКИ ==================
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 10

# Параметры Q-обучения
ALPHA = 0.1  # скорость обучения
GAMMA = 0.9  # дисконт
EPSILON = 1.0  # начальный epsilon
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
EPISODES = 1000


# ================== 2. КЛАСС ИГРЫ ==================
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Голова в центре
        self.head = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.body = [self.head]
        self.direction = (0, -1)  # вверх
        self.food = self._spawn_food()
        self.score = 0
        self.done = False
        self.steps = 0
        return self._get_state()

    def _spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in self.body:
                return pos

    def _get_state(self):
        """Возвращает бинарный вектор состояния (11 признаков)"""
        x, y = self.head
        dx, dy = self.direction

        # Определяем направления
        left_dir = (-dy, dx)
        right_dir = (dy, -dx)

        # Проверка опасности
        def is_danger(pos):
            x, y = pos
            # стена или тело
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
                return True
            return pos in self.body

        danger_straight = is_danger((x + dx, y + dy))
        danger_left = is_danger((x + left_dir[0], y + left_dir[1]))
        danger_right = is_danger((x + right_dir[0], y + right_dir[1]))

        # Направление (one-hot)
        dir_up = 1 if dy == -1 else 0
        dir_down = 1 if dy == 1 else 0
        dir_left = 1 if dx == -1 else 0
        dir_right = 1 if dx == 1 else 0

        # Где еда (относительно головы)
        fx, fy = self.food
        food_left = 1 if fx < x else 0
        food_right = 1 if fx > x else 0
        food_up = 1 if fy < y else 0
        food_down = 1 if fy > y else 0

        state = np.array([
            danger_straight,
            danger_left,
            danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down
        ], dtype=int)
        return state

    def step(self, action):
        """
        action: 0 - влево, 1 - прямо, 2 - вправо
        """
        # Преобразуем относительное действие в абсолютное направление
        dx, dy = self.direction
        left_dir = (-dy, dx)
        right_dir = (dy, -dx)

        if action == 0:  # налево
            new_dir = left_dir
        elif action == 1:  # прямо
            new_dir = self.direction
        else:  # направо
            new_dir = right_dir

        self.direction = new_dir
        x, y = self.head
        nx, ny = x + new_dir[0], y + new_dir[1]

        # Проверка столкновения
        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or (nx, ny) in self.body:
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done

        # Движение
        self.head = (nx, ny)
        self.body.insert(0, self.head)

        # Проверка еды
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._spawn_food()
        else:
            self.body.pop()
            reward = -0.1  # штраф за каждый шаг

        self.steps += 1
        # Если слишком долго нет еды - смерть
        if self.steps > 200:
            self.done = True
            reward = -5

        return self._get_state(), reward, self.done


# ================== 3. Q-ОБУЧЕНИЕ ==================
class QLearningAgent:
    def __init__(self):
        self.q_table = {}  # словарь: state_tuple -> массив из 3 значений
        self.epsilon = EPSILON

    def get_q(self, state):
        key = tuple(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(3)
        return self.q_table[key]

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # исследование
        else:
            q_vals = self.get_q(state)
            return np.argmax(q_vals)  # использование

    def update(self, state, action, reward, next_state, done):
        key = tuple(state)
        next_key = tuple(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(3)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(3)

        # Q-обновление
        best_next = np.max(self.q_table[next_key]) if not done else 0
        td_target = reward + GAMMA * best_next
        td_error = td_target - self.q_table[key][action]
        self.q_table[key][action] += ALPHA * td_error


# ================== 4. ВИЗУАЛИЗАЦИЯ (pygame) ==================
def draw_game(screen, game):
    screen.fill((0, 0, 0))
    # Еда
    fx, fy = game.food
    pygame.draw.rect(screen, (255, 0, 0), (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Змейка
    for i, (x, y) in enumerate(game.body):
        color = (0, 200, 0) if i == 0 else (0, 150, 0)
        pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()


# ================== 5. ОБУЧЕНИЕ ==================
def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game = SnakeGame()
    agent = QLearningAgent()

    for episode in range(EPISODES):
        state = game.reset()
        total_reward = 0

        while not game.done:
            # Обработка событий (чтобы окно не зависло)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Отрисовка (каждый 10-й эпизод показываем)
            if episode % 10 == 0:
                draw_game(screen, game)
                clock.tick(FPS)

        # Уменьшаем epsilon
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        if episode % 50 == 0:
            print(
                f"Эпизод {episode}, Счёт: {game.score}, Epsilon: {agent.epsilon:.3f}, Q-таблица: {len(agent.q_table)}")

    print("Обучение завершено!")
    # Демонстрация (без обучения)
    agent.epsilon = 0  # только использование
    game.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state = game._get_state()
        action = agent.act(state)
        _, _, done = game.step(action)
        draw_game(screen, game)
        clock.tick(FPS * 2)
        if done:
            game.reset()


if __name__ == "__main__":
    train()