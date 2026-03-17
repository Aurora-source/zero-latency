import discord
from discord import app_commands
from discord.ext import commands, tasks
import aiosqlite
from datetime import datetime, timedelta, timezone
import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TOKEN          = os.getenv("DISCORD_TOKEN")
GUILD_ID       = 1483104737971667117
TASK_CHANNEL_ID = 1483157371000983593
DB             = "tasks.db"
UTC            = timezone.utc

# ── Bot setup ─────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members         = True
intents.presences       = True

class TaskBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        """Called before the bot connects — best place to sync commands."""
        await init_db()
        guild = discord.Object(id=GUILD_ID)
        self.tree.copy_global_to(guild=guild)
        synced = await self.tree.sync(guild=guild)
        log.info(f"Synced {len(synced)} commands to guild {GUILD_ID}")

    async def on_ready(self):
        log.info(f"Bot online: {self.user} (ID: {self.user.id})")
        if not reminder_loop.is_running():
            reminder_loop.start()


bot = TaskBot()


# ── Database ──────────────────────────────────────────────────────────────────
async def init_db():
    async with aiosqlite.connect(DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                username      TEXT    NOT NULL,
                task          TEXT    NOT NULL,
                deadline      TEXT    NOT NULL,
                completed     INTEGER NOT NULL DEFAULT 0,
                last_reminded TEXT,
                created       TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.commit()
        # Migrate old schema if reminded column exists
        try:
            await db.execute("ALTER TABLE tasks ADD COLUMN last_reminded TEXT")
            await db.commit()
        except Exception:
            pass  # Column already exists


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_remaining(deadline: datetime) -> str:
    """Return a human-readable time-remaining string, or 'OVERDUE'."""
    delta = deadline - datetime.now(UTC)
    if delta.total_seconds() < 0:
        return "⚠️ OVERDUE"
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"⏳ {h}h {m}m left"


async def get_task_channel(bot: TaskBot) -> discord.TextChannel | None:
    return bot.get_channel(TASK_CHANNEL_ID)


# ── /assign ───────────────────────────────────────────────────────────────────
@bot.tree.command(name="assign", description="Assign a task to a member (deadline: YYYY-MM-DD HH:MM)")
@app_commands.describe(
    member   = "Who to assign the task to",
    task     = "Description of the task",
    deadline = "Deadline in YYYY-MM-DD HH:MM format (UTC)",
)
async def assign(
    interaction: discord.Interaction,
    member:   discord.Member,
    task:     str,
    deadline: str,
):
    try:
        deadline_dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        await interaction.response.send_message(
            "❌ Invalid format. Use `YYYY-MM-DD HH:MM` (e.g. `2026-03-20 14:00`)",
            ephemeral=True,
        )
        return

    if deadline_dt <= datetime.now(UTC):
        await interaction.response.send_message(
            "❌ Deadline must be in the future.",
            ephemeral=True,
        )
        return

    async with aiosqlite.connect(DB) as db:
        cursor = await db.execute(
            "INSERT INTO tasks(user_id, username, task, deadline) VALUES(?,?,?,?)",
            (member.id, member.display_name, task, deadline_dt.isoformat()),
        )
        await db.commit()
        task_id = cursor.lastrowid

    channel = await get_task_channel(bot)

    embed = discord.Embed(title="📌 New Task Assigned", color=0x00FF99)
    embed.add_field(name="Task #", value=str(task_id), inline=True)
    embed.add_field(name="Assigned to", value=member.mention, inline=True)
    embed.add_field(name="Task", value=task, inline=False)
    embed.add_field(name="Deadline", value=deadline_dt.strftime("%Y-%m-%d %H:%M UTC"), inline=True)
    embed.add_field(name="Time left", value=fmt_remaining(deadline_dt), inline=True)
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.set_footer(text=f"Assigned by {interaction.user.display_name}")

    if channel:
        await channel.send(content=member.mention, embed=embed)
        await interaction.response.send_message(
            f"✅ Task #{task_id} posted in {channel.mention}",
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(embed=embed)

    # ── DM the assigned member ────────────────────────────────────────────────
    dm_embed = discord.Embed(
        title="📌 You've been assigned a new task!",
        description=f"**{task}**",
        color=0x00FF99,
    )
    dm_embed.add_field(name="Task #",    value=str(task_id),                              inline=True)
    dm_embed.add_field(name="Deadline",  value=deadline_dt.strftime("%Y-%m-%d %H:%M UTC"), inline=True)
    dm_embed.add_field(name="Time left", value=fmt_remaining(deadline_dt),                 inline=True)
    dm_embed.add_field(name="Assigned by", value=interaction.user.display_name,            inline=False)
    dm_embed.set_footer(text="Use /done <task_id> in the server when finished.")

    try:
        await member.send(embed=dm_embed)
        log.info(f"DM sent to {member} for task #{task_id}")
    except discord.Forbidden:
        log.warning(f"Could not DM {member} — DMs may be disabled")
    except discord.HTTPException as e:
        log.warning(f"Failed to DM {member}: {e}")


# ── /tasks ────────────────────────────────────────────────────────────────────
@bot.tree.command(name="tasks", description="List all active (incomplete) tasks")
@app_commands.describe(member="Filter by a specific member (optional)")
async def tasks_list(interaction: discord.Interaction, member: discord.Member = None):
    query = "SELECT id, username, task, deadline, user_id FROM tasks WHERE completed=0"
    params = ()
    if member:
        query += " AND user_id=?"
        params = (member.id,)
    query += " ORDER BY deadline ASC"

    async with aiosqlite.connect(DB) as db:
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        await interaction.response.send_message(
            f"✅ No active tasks{f' for {member.mention}' if member else ''}.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=f"📋 Active Tasks{f' — {member.display_name}' if member else ''}",
        color=0x3498DB,
    )

    for r in rows:
        deadline = datetime.fromisoformat(r[3])
        embed.add_field(
            name=f"#{r[0]}  {r[2]}",
            value=f"👤 **{r[1]}** • {fmt_remaining(deadline)} • due {deadline.strftime('%b %d %H:%M UTC')}",
            inline=False,
        )

    embed.set_footer(text=f"{len(rows)} task(s) active")
    await interaction.response.send_message(embed=embed)


# ── /schedule ─────────────────────────────────────────────────────────────────
@bot.tree.command(name="schedule", description="View upcoming tasks sorted by deadline")
async def schedule(interaction: discord.Interaction):
    async with aiosqlite.connect(DB) as db:
        async with db.execute(
            "SELECT username, task, deadline FROM tasks WHERE completed=0 ORDER BY deadline ASC"
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        await interaction.response.send_message("📭 No scheduled tasks.", ephemeral=True)
        return

    embed = discord.Embed(title="🗓️ Task Schedule", color=0x9B59B6)
    for r in rows:
        deadline = datetime.fromisoformat(r[2])
        embed.add_field(
            name=r[1],
            value=f"👤 {r[0]}  •  🕐 {deadline.strftime('%Y-%m-%d %H:%M UTC')}  •  {fmt_remaining(deadline)}",
            inline=False,
        )

    await interaction.response.send_message(embed=embed)


# ── /done ─────────────────────────────────────────────────────────────────────
@bot.tree.command(name="done", description="Mark a task as completed")
@app_commands.describe(task_id="The task ID to mark complete")
async def done(interaction: discord.Interaction, task_id: int):
    async with aiosqlite.connect(DB) as db:
        async with db.execute(
            "SELECT task, username, completed FROM tasks WHERE id=?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        await interaction.response.send_message(f"❌ Task #{task_id} not found.", ephemeral=True)
        return

    if row[2] == 1:
        await interaction.response.send_message(
            f"ℹ️ Task #{task_id} is already completed.", ephemeral=True
        )
        return

    async with aiosqlite.connect(DB) as db:
        await db.execute("UPDATE tasks SET completed=1 WHERE id=?", (task_id,))
        await db.commit()

    embed = discord.Embed(title="✅ Task Completed", color=0x2ECC71)
    embed.add_field(name="Task #", value=str(task_id), inline=True)
    embed.add_field(name="Task", value=row[0], inline=True)
    embed.add_field(name="Completed by", value=interaction.user.mention, inline=True)
    await interaction.response.send_message(embed=embed)


# ── /delete ───────────────────────────────────────────────────────────────────
@bot.tree.command(name="delete", description="Delete a task permanently")
@app_commands.describe(task_id="The task ID to delete")
async def delete_task(interaction: discord.Interaction, task_id: int):
    async with aiosqlite.connect(DB) as db:
        async with db.execute("SELECT task FROM tasks WHERE id=?", (task_id,)) as cursor:
            row = await cursor.fetchone()

    if not row:
        await interaction.response.send_message(f"❌ Task #{task_id} not found.", ephemeral=True)
        return

    async with aiosqlite.connect(DB) as db:
        await db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        await db.commit()

    await interaction.response.send_message(
        f"🗑️ Task #{task_id} (`{row[0]}`) has been deleted.", ephemeral=True
    )


# ── /leaderboard ──────────────────────────────────────────────────────────────
@bot.tree.command(name="leaderboard", description="See the team productivity leaderboard")
async def leaderboard(interaction: discord.Interaction):
    async with aiosqlite.connect(DB) as db:
        async with db.execute("""
            SELECT username, COUNT(*) as cnt
            FROM tasks
            WHERE completed=1
            GROUP BY username
            ORDER BY cnt DESC
            LIMIT 10
        """) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        await interaction.response.send_message("📭 No completed tasks yet.", ephemeral=True)
        return

    medals = ["🥇", "🥈", "🥉"]
    embed  = discord.Embed(title="🏆 Productivity Leaderboard", color=0xF1C40F)

    for i, (username, count) in enumerate(rows):
        prefix = medals[i] if i < 3 else f"#{i+1}"
        embed.add_field(
            name=f"{prefix}  {username}",
            value=f"{count} task{'s' if count != 1 else ''} completed",
            inline=False,
        )

    embed.set_footer(text="Top 10 — all time")
    await interaction.response.send_message(embed=embed)


# ── /mystats ──────────────────────────────────────────────────────────────────
@bot.tree.command(name="mystats", description="See your personal task stats")
async def mystats(interaction: discord.Interaction):
    uid = interaction.user.id
    async with aiosqlite.connect(DB) as db:
        async with db.execute(
            "SELECT completed, COUNT(*) FROM tasks WHERE user_id=? GROUP BY completed", (uid,)
        ) as cursor:
            rows = await cursor.fetchall()

    done_count   = next((r[1] for r in rows if r[0] == 1), 0)
    active_count = next((r[1] for r in rows if r[0] == 0), 0)

    embed = discord.Embed(
        title=f"📊 Stats — {interaction.user.display_name}",
        color=0x1ABC9C,
    )
    embed.set_thumbnail(url=interaction.user.display_avatar.url)
    embed.add_field(name="✅ Completed", value=str(done_count),   inline=True)
    embed.add_field(name="🔄 Active",    value=str(active_count), inline=True)
    embed.add_field(name="📦 Total",     value=str(done_count + active_count), inline=True)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── Reminder loop ─────────────────────────────────────────────────────────────
# Sends reminders every 4 hours, but only between 10:00 AM – 12:00 AM IST
# Quiet hours: 12:00 AM – 10:00 AM IST (i.e. no messages at night/early morning)
IST          = timezone(timedelta(hours=5, minutes=30))
QUIET_START  = 0   # 12:00 AM IST
QUIET_END    = 10  # 10:00 AM IST
REMIND_EVERY = timedelta(hours=4)

def is_quiet_hours(now: datetime) -> bool:
    """Returns True if current IST time is inside the quiet window (12AM–10AM IST)."""
    now_ist = now.astimezone(IST)
    return QUIET_START <= now_ist.hour < QUIET_END

@tasks.loop(minutes=30)
async def reminder_loop():
    now = datetime.now(UTC)

    # Don't send anything during quiet hours
    if is_quiet_hours(now):
        log.info(f"Reminder loop skipped — quiet hours ({now.strftime('%H:%M')} UTC)")
        return

    channel = await get_task_channel(bot)

    async with aiosqlite.connect(DB) as db:
        async with db.execute(
            "SELECT id, user_id, task, deadline, last_reminded FROM tasks WHERE completed=0"
        ) as cursor:
            rows = await cursor.fetchall()

    for task_id, user_id, task_text, deadline_str, last_reminded_str in rows:
        deadline   = datetime.fromisoformat(deadline_str)
        time_left  = deadline - now

        # Skip tasks that haven't started or are already past (more than 24h overdue)
        if time_left < timedelta(hours=-24):
            continue

        # Check if enough time has passed since last reminder
        if last_reminded_str:
            last_reminded = datetime.fromisoformat(last_reminded_str)
            if now - last_reminded < REMIND_EVERY:
                continue  # Too soon to remind again

        # Build the embed
        if time_left.total_seconds() < 0:
            urgency_color = 0xFF0000   # red — overdue
            urgency_label = "🚨 OVERDUE"
        elif time_left <= timedelta(hours=2):
            urgency_color = 0xFF6B35   # orange — very soon
            urgency_label = "🔥 Due very soon!"
        elif time_left <= timedelta(hours=12):
            urgency_color = 0xF1C40F   # yellow — today
            urgency_label = "⚠️ Due today"
        else:
            urgency_color = 0xE74C3C   # default red
            urgency_label = "⏰ Upcoming deadline"

        try:
            user = await bot.fetch_user(user_id)
        except discord.NotFound:
            continue

        embed = discord.Embed(title=f"⏰ Task Reminder — {urgency_label}", color=urgency_color)
        embed.add_field(name="Task #",    value=str(task_id),                               inline=True)
        embed.add_field(name="Task",      value=task_text,                                   inline=True)
        embed.add_field(name="Due",       value=deadline.strftime("%Y-%m-%d %H:%M UTC"),    inline=False)
        embed.add_field(name="Time left", value=fmt_remaining(deadline),                     inline=True)
        embed.set_footer(text="Reminders sent every 4 hours • Quiet hours: 12AM–10AM IST")

        if channel:
            await channel.send(content=user.mention, embed=embed)

        try:
            await user.send(embed=embed)
        except discord.Forbidden:
            pass  # DMs disabled
        except discord.HTTPException as e:
            log.warning(f"Failed to DM {user}: {e}")

        async with aiosqlite.connect(DB) as db:
            await db.execute(
                "UPDATE tasks SET last_reminded=? WHERE id=?",
                (now.isoformat(), task_id)
            )
            await db.commit()

        log.info(f"Reminded {user} about task #{task_id} ({fmt_remaining(deadline)})")


@reminder_loop.before_loop
async def before_reminder():
    await bot.wait_until_ready()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set!")
    bot.run(TOKEN, log_handler=None)
