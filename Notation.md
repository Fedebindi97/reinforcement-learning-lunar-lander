$$Q(s_t,a_t) = r_t + E(r_{t+1} + r_{t+2}...)$$
$$Q(s_t,a_t) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}...$$
$$Q(s_t,a_t) = r_t + \gamma Q(s_{t+1},a_{t+1})$$
$$Q(s_t,a_t) = r_t + \gamma \ \underset{a}\max Q(s_{t+1},a)$$
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t + \gamma \ \underset{a}\max Q(s_{t+1},a) - Q(s_t,a_t)]$$
$$
\left\{
\begin{array}{l}
w.p. \ 1-\epsilon \rightarrow choose \ a \ s.t. \  \underset{a}\max Q(s_{t+1},a) \\
w.p. \ \epsilon \rightarrow choose \ random \ action
\end{array}
\right.
$$
$$Q(s) = \underset{a}\max (R(s,a) + \gamma Q(s^{\prime}))$$