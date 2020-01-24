Q[s, a] = Q[s, a] + alpha * (reward +
        gamma * Q[s_, a_] - Q[s, a])
