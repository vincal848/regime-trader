# Running it on a Windows PC

This is the paper-trading setup on one Windows 11 machine:

- **IB Gateway**, kept logged in by **IBC**;
- the **trader**, as a Windows service via NSSM;
- the **nightly review**, the **watchdog** and the monthly **refit**, as
  Task Scheduler jobs;
- the **dashboard**, on demand.

Times below are US Eastern (ET). If your PC uses another time zone, convert
them when you create the scheduled tasks.

## 1. Accounts and data (you)

1. **IBKR paper account.** In Client Portal, under Settings → Paper Trading
   Account, note the paper username and the account ID (it starts with
   `DU`).
2. **Market data.** Paper accounts share the live account's market-data
   subscriptions. SPY hourly bars need US equity data. Without it, IB
   returns delayed or no data, and the trader stays flat (stale data).
3. **Telegram (optional).** Create a bot with @BotFather and note its
   token. Message the bot once, then read your chat ID from
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser. The
   trader only ever *sends*.
4. **Anthropic (optional).** Create an API key for the nightly review.
   Set a spend limit in the Anthropic Console as well as the code's own
   monthly cap.

## 2. Install

```powershell
cd D:\
git clone <this repo> regime-trader
cd regime-trader
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[ibkr,llm,dashboard,demo]"
copy .env.example .env
notepad .env
```

In `.env`:
- set `IB_ACCOUNT` to the `DU…` paper account ID;
- keep `IB_PORT=4002`, IB Gateway's paper port;
- add the Telegram token and chat ID and the Anthropic key if you use them;
- set `LLM_MONTHLY_BUDGET_USD`.

Never commit `.env`; it is gitignored. The trader refuses any account that
doesn't start with `DU`, and the live ports 4001 and 7496.

## 3. IB Gateway and IBC

1. Install **IB Gateway (stable)** from IBKR.
2. Install **IBC** (github.com/IbcAlpha/IBC) and follow its Windows user
   guide. In IBC's `config.ini`:
   - `TradingMode=paper`;
   - your paper username and password, which **you** type into IBC's own
     file (this repo never reads it, and Claude never asks for it);
   - `AcceptIncomingConnectionAction=accept`;
   - `ReadOnlyApi=no`.

   Check each setting name against the IBC user guide for your version.
3. In IB Gateway, under Configure → Settings → API → Settings:
   - socket port **4002**;
   - "Allow connections from localhost only" on;
   - trusted IP `127.0.0.1`;
   - "Read-Only API" off, so paper orders can be placed.
4. Use IBC's `StartGateway.bat` as a Task Scheduler job "At log on", so
   Gateway starts with the machine. IB Gateway logs out daily; let IBC
   handle the restart and re-login.

Paper logins usually don't ask for two-factor authentication. If yours
does, IBC can't fully automate it. Plan to re-authenticate on your phone
after each weekly reset.

## 4. First run

```powershell
.venv\Scripts\regime-trader fetch --source ibkr --years 6
.venv\Scripts\regime-trader fit
.venv\Scripts\regime-trader backtest
```

`fetch` needs IB Gateway running. `backtest` walks forward from 2021 with
the most recent 12 months locked away. It exits with code 1 when the gates
fail, which is a result, not an error.

**Read the backtest report before you trade.** If the gates fail, paper
trading is still useful for checking the plumbing, but the report should
set your expectations.

## 5. Windows settings the trader needs

- **Never sleep while plugged in:**
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  powercfg /hibernate off
  ```
- **No update restarts during market hours.** Set Settings → Windows
  Update → Advanced options → Active hours to cover at least 08:30–17:00
  ET.
- **Automatic login,** so IB Gateway can start after a power cut: use
  Sysinternals **Autologon**, which stores the password encrypted in LSA
  secrets, or `netplwiz`. Lock the screen (Win+L) after it logs in.
- **BIOS:** "Restore on AC power loss" set to Power On, if your machine
  has it.
- **A UPS** is worth it. A power cut while in a position leaves the
  position unmanaged, and the kill switch can only act after reconnecting.

## 6. The trader as a service (NSSM)

```powershell
winget install NSSM.NSSM
mkdir D:\regime-trader\logs
nssm install RegimeTrader D:\regime-trader\.venv\Scripts\regime-trader.exe "--root D:\regime-trader live"
nssm set RegimeTrader AppDirectory D:\regime-trader
nssm set RegimeTrader AppStdout D:\regime-trader\logs\live.log
nssm set RegimeTrader AppStderr D:\regime-trader\logs\live.err
nssm set RegimeTrader AppExit Default Restart
nssm set RegimeTrader AppRestartDelay 30000
nssm set RegimeTrader Start SERVICE_DELAYED_AUTO_START
nssm start RegimeTrader
```

- The service restarts 30 s after any crash.
- The trader resumes from `live_state.json`; a restart never decides the
  same bar twice.
- A kill switch engaged before the crash stays engaged. It is a file in
  `control/`.

## 7. Scheduled jobs (Task Scheduler)

```powershell
$exe = "D:\regime-trader\.venv\Scripts\regime-trader.exe"
# nightly report and Claude review, weekdays after the close
schtasks /Create /TN "RegimeTrader\Nightly" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:30 /TR "$exe --root D:\regime-trader nightly"
# watchdog every 15 minutes (it only alerts during the session)
schtasks /Create /TN "RegimeTrader\Watchdog" /SC MINUTE /MO 15 /TR "$exe --root D:\regime-trader watchdog"
# monthly refit on the first Saturday, then restart the trader so it loads the new fit
schtasks /Create /TN "RegimeTrader\Refit" /SC MONTHLY /MO FIRST /D SAT /ST 10:00 /TR "cmd /c $exe --root D:\regime-trader fetch && $exe --root D:\regime-trader fit && nssm restart RegimeTrader"
```

The refit prints a drift line and archives the previous fit. If drift
fires, you get an alert. Review the new fit before Monday: if it looks
wrong, restore the archived file as `models\fit.json`.

## 8. Day-to-day

| Task | Command |
|---|---|
| Dashboard | `regime-trader --root D:\regime-trader dashboard` |
| Stop trading now (flattens at the next bar, then stays flat) | `regime-trader --root D:\regime-trader kill --reason "..."` |
| Resume after a kill | `regime-trader --root D:\regime-trader kill --reset` |
| Allow orders over $25,000 for an hour | `regime-trader --root D:\regime-trader approve --minutes 60` |
| Today's report | `regime-trader --root D:\regime-trader report` |
| Review Claude's proposals | open `proposals\<date>\evaluation.md` |

To flatten *immediately*, close the position in IB Gateway's companion
TWS, or in the IBKR mobile app. The trader treats the broker's position as
the truth at its next bar.

## 9. Why not Vercel, or a serverless host?

The trader needs a long-running process holding a socket to IB Gateway,
and IB Gateway itself is a Java desktop application that must stay logged
in. Serverless platforms run short-lived functions, so they can host
neither. A small always-on VPS (Windows or Linux) would remove the
home-power and home-internet risks. The setup there would be the same,
with IBC and a service manager.
