import jax
import jax.numpy as jnp


def random_piston_positions(key, n, r):
    def sample(key):
        key, k = jax.random.split(key)
        x = jax.random.uniform(k, (3,), minval=-1.0 + r, maxval=1.0 - r)
        return key, x

    def body_for(i, args):
        key, pos = args

        def cond(args):
            key, x = args
            test = (jnp.linalg.norm(x[:2]) < 1.0 - r) & jnp.all(jnp.linalg.norm(pos - x, axis=1) > 2 * r)
            return ~test

        def body(args):
            key, x = args
            return sample(key)

        key, x = sample(key)
        key, x = jax.lax.while_loop(cond, body, (key, x))
        return key, pos.at[i].set(x)

    pos = 2.0 * jnp.ones((n, 3))
    key, pos = jax.lax.fori_loop(0, n, body_for, (key, pos))

    return key, pos


def init_piston(key, n, r):
    walls = {
        'x': jnp.array([
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]),
        'v': jnp.zeros((2, 3)),
        'j': jnp.array([
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]),
        'k': jnp.array([
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        'm': jnp.inf * jnp.ones(2),
    }

    cylinders = {
        'x': jnp.array([[0.0, 0.0, 0.0]]),
        'v': jnp.array([[0.0, 0.0, 0.0]]),
        'j': jnp.array([[0.0, 0.0, 1.0]]),
        'r': jnp.array([1.0]),
        'm': jnp.array([jnp.inf]),
    }

    key, x = random_piston_positions(key, n, r)

    v = jax.random.normal(key, (n, 3))

    balls = {
        'x': x,
        'v': v,
        'r': r * jnp.ones(n),
        'm': jnp.ones(n),
    }

    state = {
        'balls': balls,
        'walls': walls,
        'cylinders': cylinders,
    }

    return state


# init_piston_j = jax.jit(init_piston, static_argnums=1)
# init_piston_vj = jax.jit(jax.vmap(init_piston, (0, None, None), 0), static_argnums=1)


def energy(state):
    m = state['balls']['m']
    v = state['balls']['v']
    return 0.5 * jnp.sum(m * jnp.sum(v**2, 1))


# energy_j = jax.jit(energy)
# energy_vj = jax.jit(jax.vmap(energy, (0,), 0))
