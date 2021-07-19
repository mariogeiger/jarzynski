import jax
import jax.numpy as jnp


def narrow(dim, index):
    return (slice(None, None, None),) * dim + (index,)


def imap(f, in_axes, out_axes=0):
    def g(*args):
        ns = [a.shape[i] for i, a in zip(in_axes, args) if i is not None]
        n = ns[0]
        assert len(set(ns)) == 1

        b = [a if i is None else a[narrow(i, 0)] for i, a in zip(in_axes, args)]
        outs = jax.tree_map(lambda x: jnp.empty(x.shape[:out_axes] + (n,) + x.shape[out_axes:]), f(*b))

        def body(i, outs):
            b = [a if in_axis is None else a[narrow(in_axis, i)] for in_axis, a in zip(in_axes, args)]
            return jax.tree_multimap(
                lambda out, fi: out.at[narrow(out_axes, i)].set(fi),
                outs, f(*b)
            )

        return jax.lax.fori_loop(0, n, body, outs)

    return g
