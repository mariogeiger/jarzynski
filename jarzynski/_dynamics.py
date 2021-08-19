import jax.numpy as jnp
import jax


def ball_ball_collision_time(a, b):
    x = b['x'] - a['x']
    v = b['v'] - a['v']
    r = a['r'] + b['r']
    xv = jnp.sum(x * v)

    def f(_):
        vv = jnp.sum(v * v)
        dis = xv * xv - vv * (jnp.sum(x * x) - r * r)

        return jax.lax.cond(
            dis > 0.0,
            lambda _: (-xv - jnp.sqrt(dis)) / vv,
            lambda _: jnp.inf,
            None
        )

    return jax.lax.cond(xv < 0.0, f, lambda _: jnp.inf, None)


def ball_wall_collision_time(a, w):
    x = a['x'] - w['x']
    v = a['v'] - w['v']
    n = jnp.cross(w['j'], w['k'])
    njck = jnp.linalg.norm(n)
    n = n / njck
    vn = jnp.sum(v * n)
    xn = jnp.sum(x * n)

    return jax.lax.cond(
        xn * vn < 0.0,
        lambda _: -xn / vn - jnp.abs(a['r'] / vn),
        lambda _: jnp.inf,
        None
    )


def ball_cylinder_collision_time(a, c):
    x = a['x'] - c['x']
    v = a['v'] - c['v']
    j = c['j']
    jj = jnp.sum(j * j)

    xj = jnp.sum(x * j)
    vj = jnp.sum(v * j)

    xx = jnp.sum(x * x) - xj**2 / jj
    xv = jnp.sum(x * v) - (xj * vj) / jj
    vv = jnp.sum(v * v) - vj**2 / jj
    del x, v, j, jj

    s = jnp.where(xx < c['r']**2, -1, 1)
    dis = xv**2 - vv * (xx - (c['r'] + s * a['r'])**2)

    def sol(sign):
        return (-xv + sign * jnp.sqrt(dis)) / vv

    def outside(_):
        return jax.lax.cond(xv > 0.0, lambda _: jnp.inf, sol, -1)

    def exists(_):
        return jax.lax.cond(xx < c['r']**2, sol, outside, 1)

    return jax.lax.cond(dis > 0.0, exists, lambda _: jnp.inf, None)


def collision(n, va, ma, vb, mb):
    def col(pa):
        n, va, ma, vb, mb = pa

        fv = jax.lax.cond(
            jnp.isinf(mb),
            lambda _: vb,
            lambda _: (ma * va + mb * vb) / (ma + mb),
            None
        )

        # go in rest frame
        va = va - fv
        vb = vb - fv

        # collision
        va = va - 2 * jnp.sum(n * va) * n / jnp.sum(n * n)
        vb = -ma / mb * va

        # go back in original frame
        va = va + fv
        vb = vb + fv
        return va, vb

    def col_a_inf(pa):
        n, va, ma, vb, mb = pa
        vb, va = col((n, vb, mb, va, ma))
        return va, vb

    return jax.lax.cond(jnp.isinf(ma), col_a_inf, col, (n, va, ma, vb, mb))


def _min(x, initial):
    x = x.flatten()
    if x.shape[0] == 0:
        return initial, 0
    i = jnp.argmin(x)
    return x[i], i


def update(dt, state):
    balls, walls, cylinders = state['balls'], state['walls'], state['cylinders']

    bb = jax.vmap(jax.vmap(ball_ball_collision_time, (None, 0), 0), (0, None), 0)(balls, balls)
    bc = jax.vmap(jax.vmap(ball_cylinder_collision_time, (None, 0), 0), (0, None), 0)(balls, cylinders)
    bw = jax.vmap(jax.vmap(ball_wall_collision_time, (None, 0), 0), (0, None), 0)(balls, walls)

    bb = jnp.maximum(bb, 0.0)
    bc = jnp.maximum(bc, 0.0)
    bw = jnp.maximum(bw, 0.0)

    bb_min, bb_amin = _min(bb, jnp.inf)
    bc_min, bc_amin = _min(bc, jnp.inf)
    bw_min, bw_amin = _min(bw, jnp.inf)

    tcol = jnp.min(jnp.array([bb_min, bc_min, bw_min]))
    tf = jnp.minimum(dt, tcol)

    balls['x'] = balls['x'] + tf * balls['v']
    walls['x'] = walls['x'] + tf * walls['v']
    cylinders['x'] = cylinders['x'] + tf * cylinders['v']

    def fun_bb(_):
        ia = bb_amin // bb.shape[1]
        ib = bb_amin % bb.shape[1]
        n = balls['x'][ia] - balls['x'][ib]

        va, vb = collision(n, balls['v'][ia], balls['m'][ia], balls['v'][ib], balls['m'][ib])
        v = balls['v']
        v = v.at[ia].set(va)
        v = v.at[ib].set(vb)
        return v, walls['v'], cylinders['v'], 0.0

    def fun_bc(_):
        ia = bc_amin // bc.shape[1]
        ic = bc_amin % bc.shape[1]
        x = balls['x'][ia] - cylinders['x'][ic]
        j = cylinders['j'][ic]
        n = x - j * jnp.sum(x * j) / jnp.sum(j * j)

        va, vc = collision(n, balls['v'][ia], balls['m'][ia], cylinders['v'][ic], cylinders['m'][ic])
        return (
            balls['v'].at[ia].set(va),
            walls['v'],
            cylinders['v'].at[ic].set(vc),
            0.0
        )

    def fun_bw(_):
        ib = bw_amin // bw.shape[1]
        iw = bw_amin % bw.shape[1]
        n = jnp.cross(walls['j'][iw], walls['k'][iw])
        vb0, mb, vw, mw = balls['v'][ib], balls['m'][ib], walls['v'][iw], walls['m'][iw]

        vb1, vw1 = collision(n, vb0, mb, vw, mw)
        work = mb * jnp.sum((vb1 - vb0) * vw)

        return (
            balls['v'].at[ib].set(vb1),
            walls['v'].at[iw].set(vw1),
            cylinders['v'],
            work
        )

    def bar(_):
        def foo(_):
            if cylinders['x'].shape[0] == 0:
                return fun_bw(None)
            if walls['x'].shape[0] == 0:
                return fun_bc(None)
            return jax.lax.cond(bw_min == tcol, fun_bw, fun_bc, None)

        return jax.lax.cond(bb_min == tcol, fun_bb, foo, None)

    balls['v'], walls['v'], cylinders['v'], work = jax.lax.cond(
        tf == tcol,
        bar,
        lambda _: (balls['v'], walls['v'], cylinders['v'], 0.0),
        None
    )

    state = {
        'balls': balls,
        'walls': walls,
        'cylinders': cylinders,
    }
    return tf, state, work


def forward_n(n, state):
    def body(i, args):
        t, state = args
        dt, state, _ = update(jnp.inf, state)
        return t + dt, state

    return jax.lax.fori_loop(0, n, body, (0.0, state))


def forward(t, state):
    def cond(val):
        t0, state, n, work = val
        return t0 < t

    def body(val):
        t0, state, n, work = val
        dt, state, w = update(t - t0, state)
        return t0 + dt, state, n + 1, work + w

    t, state, n, work = jax.lax.while_loop(cond, body, (jnp.array(0.0), state, jnp.array(0), jnp.array(0.0)))

    return n, state, work
