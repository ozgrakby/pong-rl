from core.env import PongEnv
from ai.agent import Agent
import config, os, datetime, time

def train():
    now = datetime.datetime.now()
    log_filename = f"logs/training_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    baslangic_zamani = time.time()
    f = open(log_filename, "w", encoding = "utf-8")

    env = PongEnv()
    agent_l = Agent()
    agent_r = Agent()

    if os.path.exists(config.AGENT1_PATH):
        print("Birinci ajan yükleniyor...")
        agent_l.load(config.AGENT1_PATH)
    
    if os.path.exists(config.AGENT2_PATH):
        print("İkinci ajan yükleniyor...")
        agent_r.load(config.AGENT2_PATH)

    total_steps = 0
    episode = 0
    try:
        while True:
            episode += 1
            state = env.reset()
            done = False
            episode_reward_l = 0
            episode_reward_r = 0

            while not done:
                act_l = agent_l.act(state)
                act_r = agent_r.act(state)

                next_state, (reward_l, reward_r), done = env.step(act_l, act_r)

                agent_l.remember(state, act_l, reward_l, next_state, done)
                agent_r.remember(state, act_r, reward_r, next_state, done)

                if total_steps % 4 == 0:
                    agent_l.learn()
                    agent_r.learn()

                if total_steps % config.TARGET_UPDATE_FREQ == 0:
                    agent_l.update_target()
                    agent_r.update_target()

                state = next_state
                total_steps += 1
                episode_reward_l += reward_l
                episode_reward_r += reward_r

            agent_l.decay_epsilon()
            agent_r.decay_epsilon()
            
            if episode % 10 == 0:
                str = f"Ep: {episode} | Steps: {total_steps} | AvgSPE: {(total_steps/episode):.1f} |" + f"Epsilon: {agent_l.epsilon:.4f} | " + f"Skor: {episode_reward_l:.1f} / {episode_reward_r:.1f}"
                f.write(str + "\n")
                f.flush()
                print(str)
            
            if episode % config.SAVE_INTERVAL == 0:
                    agent_l.save(config.AGENT1_PATH)
                    agent_r.save(config.AGENT2_PATH)
                    print(f"Modeller kaydedildi: Episode {episode}")
    
    except KeyboardInterrupt:
        print("\nEğitim kullanıcı tarafından durduruldu.")
        agent_l.save(config.AGENT1_PATH)
        agent_r.save(config.AGENT2_PATH)
        print("Son durum kaydedildi.")
        env.close()
        bitis_zamani = time.time()
        gecen_sure = bitis_zamani - baslangic_zamani

        dakika = int(gecen_sure // 60)
        saniye = int(gecen_sure % 60)

        f.write(f"Eğitim {dakika} dakika {saniye} saniye sürdü")
        f.flush()
        f.close()

def test():
    env = PongEnv(render_mode=True)
    
    agent_left = Agent(mode='test')
    agent_right = Agent(mode='test')
    
    # Modelleri Yükle (Zorunlu)
    if os.path.exists(config.AGENT1_PATH):
        agent_left.load(config.AGENT1_PATH)
        print("Birinci Ajan yüklendi.")
    else:
        print("UYARI: Birinci Ajan dosyası bulunamadı!")

    if os.path.exists(config.AGENT2_PATH):
        agent_right.load(config.AGENT2_PATH)
        print("İkinci Ajan yüklendi.")
    else:
        print("UYARI: İkinci Ajan dosyası bulunamadı!")

    clock = env.clock

    running = True
    while running:
        state = env.reset()
        done = False
        
        while not done:
            for event in env.pg.event.get():
                if event.type == env.pg.QUIT:
                    running = False
                    done = True
            
            if not running: break

            action_l = agent_left.act(state, training=False)
            action_r = agent_right.act(state, training=False)
            
            next_state, _, done = env.step(action_l, action_r)
            
            state = next_state
            
            clock.tick(config.FPS)

    env.close()

if __name__ == "__main__":
    print("1. Modeli Eğit (Train)")
    print("2. Modeli İzle (Test/Play)")
    
    choice = input("Seçiminiz (1 veya 2): ")
    
    if choice == "1":
        train()
    elif choice == "2":
        test()
    else:
        print("Geçersiz seçim.")