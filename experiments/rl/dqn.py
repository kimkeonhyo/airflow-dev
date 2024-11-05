import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from generate_data import generate_similar_data

# 경험 재생을 위한 메모리 구조 정의
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class LungCancerEnvironment:
    def __init__(self, initial_data, generate_data_fn):
        self.generate_data_fn = generate_data_fn
        self.initial_data = initial_data
        self.reset()
    
    def reset(self):
        # 새로운 데이터 생성
        self.current_data = self.generate_data_fn(self.initial_data, n_samples=1).iloc[0]
        # LUNG_CANCER 열을 제외한 특성을 상태로 변환
        state = self._get_state_from_data(self.current_data)
        return state
    
    def _get_state_from_data(self, data):
        # LUNG_CANCER를 제외한 모든 특성을 수치형으로 변환
        state_data = data.drop('LUNG_CANCER')
        # Gender를 이진값으로 변환
        state_data['GENDER'] = 1 if state_data['GENDER'] == 'M' else 0
        return torch.FloatTensor(state_data.astype(float).values)
    
    def step(self, action):
        # action: 0 (No cancer) or 1 (Cancer)
        actual_result = 1 if self.current_data['LUNG_CANCER'] == 'YES' else 0
        
        # 보상 계산
        if action == actual_result:
            reward = 1.0  # 정확한 예측
        else:
            reward = -1.0  # 잘못된 예측
            
        # 다음 상태를 위한 새로운 데이터 생성
        self.current_data = self.generate_data_fn(self.initial_data, n_samples=1).iloc[0]
        next_state = self._get_state_from_data(self.current_data)
        
        done = False  # 에피소드는 계속 진행
        return next_state, reward, done

class LungCancerAgent:
    def __init__(self, state_size, hidden_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = 64
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].item()
        else:
            return random.randint(0, 1)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, experiences):
        if len(experiences) < self.batch_size:
            return
        
        batch = random.sample(experiences, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_rl_model(data, num_episodes=100):
    # 환경과 에이전트 초기화
    env = LungCancerEnvironment(data, generate_similar_data)
    state_size = len(data.columns) - 1  # LUNG_CANCER 열 제외
    agent = LungCancerAgent(state_size)
    
    # 학습 메트릭스
    rewards_history = []
    accuracy_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        correct_predictions = 0
        total_predictions = 0
        
        # 각 에피소드에서 여러 스텝 수행
        for step in range(100):  # 각 에피소드당 100개의 예측 수행
            # 행동 선택
            action = agent.select_action(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, done = env.step(action)
            
            # 경험 저장
            agent.memory.push(Experience(state, action, reward, next_state, done))
            
            # 모델 학습
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train(agent.memory.memory)
            
            # 메트릭스 업데이트
            episode_reward += reward
            if reward > 0:
                correct_predictions += 1
            total_predictions += 1
            
            # 다음 상태로 이동
            state = next_state
            
            if done:
                break
        
        # 에피소드 종료 후 업데이트
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 메트릭스 저장
        accuracy = correct_predictions / total_predictions
        rewards_history.append(episode_reward)
        accuracy_history.append(accuracy)
        
        # 진행 상황 출력
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Reward: {episode_reward:.2f}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
            print("------------------------")
    
    return agent, rewards_history, accuracy_history

# 학습 실행
if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')
    
    # 모델 학습
    agent, rewards, accuracies = train_rl_model(data)
    
    # 결과 시각화
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Episode Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # 모델 저장
    torch.save(agent.policy_net.state_dict(), 'models/lung_cancer_rl_model.pth')