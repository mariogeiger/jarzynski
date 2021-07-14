import jax
import jax.numpy as jnp


def narrow(x, dim, index):
    return x[(Ellipsis,) * dim + (index,) + (Ellipsis,) * (len(x.shape) - dim - 1)]


def imap(f, in_axes, out_axes=0):
    def g(*args):
        ns = [a.shape[i] for i, a in zip(in_axes, args) if i is not None]
        n = ns[0]
        assert len(set(ns)) == 1

        b = [a if i is None else narrow(a, i, 0) for i, a in zip(in_axes, args)]
        outs = jax.tree_map(lambda x: jnp.empty(x.shape[:out_axes] + (n,) + x.shape[out_axes:]), f(*b))

        def body(j, outs):
            b = [a if i is None else narrow(a, i, j) for i, a in zip(in_axes, args)]
            out = f(*b)
            return jax.tree_multimap(lambda x, y: x.at[(Ellipsis,) * out_axes + (j,) + (Ellipsis,) * (len(x.shape)-out_axes-1)].set(out), outs, out)

        return jax.lax.fori_loop(0, n, body, outs)

    return g
