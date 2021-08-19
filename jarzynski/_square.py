import jax
import jax.numpy as jnp


def random_square_positions(key, n, r):
    def sample(key):
        key, k = jax.random.split(key)
        x = jax.random.uniform(k, (2,), minval=-1.0 + r, maxval=1.0 - r)
        x = jnp.concatenate([x, jnp.ones((1,))])
        return key, x

    def body_for(i, args):
        key, pos = args

        def cond(args):
            key, x = args
            return ~jnp.all(jnp.linalg.norm(pos - x, axis=1) > 2 * r)

        def body(args):
            key, x = args
            return sample(key)

        key, x = sample(key)
        key, x = jax.lax.while_loop(cond, body, (key, x))
        return key, pos.at[i].set(x)

    pos = 2.0 * jnp.ones((n, 3))
    key, pos = jax.lax.fori_loop(0, n, body_for, (key, pos))

    return key, pos


def init_square(key, n, r):
    walls = {
        'x': jnp.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]),
        'v': jnp.zeros((4, 3)),
        'j': jnp.array([
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        'k': jnp.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ]),
        'm': jnp.inf * jnp.ones(4),
    }

    cylinders = {
        'x': jnp.ones((0, 3)),
        'v': jnp.ones((0, 3)),
        'j': jnp.ones((0, 3)),
        'r': jnp.ones((0, )),
        'm': jnp.ones((0, )),
    }

    key, x = random_square_positions(key, n, r)

    v = jax.random.normal(key, (n, 3))
    v = v.at[:, 2].set(0.0)

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
