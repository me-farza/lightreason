# Vicena Lightreason demo

Use a fresh demo workspace and select **Gemini · local learned memory**. The notes below describe a fictional lab workflow. In this mode, Gemini answers using the current chat and, when local memory finds a match, a short decoded readout. During Dream, Gemini can propose questions and organize note excerpts. The local 3B memory policy runs the optimizer and validation.

## Day 1 · Teach two procedures

Send these as **two separate chat messages** so the wiki can track each procedure's revisions. **Save for Dream** is on by default for this chat. Uncheck it before sending any message you want to leave out of the Dream queue.

```text
For baseline drift, use a fresh reference first.
```

```text
For visual scoring, blind the scorer first.
```

Open **Architecture** in the top navigation to show where chat, Dream and the wiki connect. Return to **Chat**, open **Dream** and click **Run Dream**. Under **Questions from your notes**, show a question paired with the action from the note. Under **Local 3B training and validation**, expand a procedure to see the optimizer steps, sampled mean reward, and policy entropy. Mean reward averages the sampled recall and abstention decisions scored by the harness. Entropy shows how uncertain those sampled decisions were. Look for **Saved** once a candidate passes validation. A **Held** candidate leaves the active memory as it was before that update.

When both procedures show **Saved**, click **Morning**. Send the following checks with **Save for Dream** unchecked. Open **Inspect memory** under each reply.

| Ask in chat | Expected in Inspect memory |
| --- | --- |
| For the baseline drift, what should I set or do? | The readout says to use a fresh reference first. |
| What settings does the unrelated reference procedure use? | No local memory was recalled. Gemini may still answer the question. |

## Day 2 · Correct one procedure and keep the other

Send this correction with **Save for Dream** checked.

```text
Correction: for baseline drift, use a sealed blank first.
```

Click **Dream**, then **Run Dream**. Show the held-out check for the corrected rule and the retention check for the other lesson. When the correction shows **Saved**, return to **Morning**. The fresh chat lets you test recall without including the teaching turns. For isolated checks, use **New chat** between questions and uncheck **Save for Dream** before each send.

| Ask in chat | Expected in Inspect memory |
| --- | --- |
| For the baseline drift, what should I set or do? | Use a sealed blank first. |
| For the visual scoring, what should I set or do? | Blind the scorer first. |
| What settings does the unrelated reference procedure use? | No local memory was recalled. |

Use these known held-out wordings for the main demo. If you want to show a new scenario, try “A new spectrometer shows baseline drift between runs. What check should I do first?” and inspect whether the learned memory was recalled.

## Open the saved evidence

For these checks, uncheck **Save for Dream** before sending. In chat, ask **Search the wiki for baseline drift**. Open **Inspect memory** to see the local tool step and its source link. You can also open **Memory wiki** and search for `baseline drift`. Open the corrected note to show **Original text** and **Note history**. The earlier note remains available as an older version. In chat, “Which memories do you have?” reports the local inventory. With only the three teaching turns saved, expect two active notes, one older version, and zero waiting for Dream.

To demonstrate restart persistence, have the operator restart Vicena with the same database and Dream state. Open **New chat** and repeat the Day 2 checks. Compare the decoded readouts under **Inspect memory**.
