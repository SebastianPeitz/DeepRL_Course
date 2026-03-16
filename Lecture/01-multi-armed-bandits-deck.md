---
subtitle:    Multi-armed bandits
feedback:
  deck-id:  'deeprl-multi-armed-bandits'
...


# Example: Route planning (OH16 to Hansaplatz by car)

::: platzhalter

::: columns-7-3
![Google Maps (go [here](https://maps.app.goo.gl/ixKDSXT6bwzfa6Y9A) for the live version; do you get the same numbers?)](images/MapsDortmund.png){ width=900px }

::: small
::: incremental
- Let's assume we do not have access to travel time estimates
- Which route should I take to minimize my travel time?
- Let's say we can guess the time of one route fairly well
  - should we always take this one?
  - or try something else and see if we can get better?
- this is known as the **exploration-exploitation dilemma**
- the route pickig problem is one example of a **multi-armed bandit**
:::
:::

:::

:::
<!-- https://www.google.com/maps/dir/Otto-Hahn-Stra%C3%9Fe+16,+44227+Dortmund/Hansapl.,+44137+Dortmund-Innenstadt-West/@51.5005743,7.4251556,14z/data=!4m14!4m13!1m5!1m1!1s0x47b918ff9cf944b1:0x3f36034c67e5dc28!2m2!1d7.4051747!2d51.489565!1m5!1m1!1s0x47b919e1942e6f9d:0xbf57c39d87666683!2m2!1d7.4652885!2d51.512955!3e0?entry=ttu&g_ep=EgoyMDI2MDMwMi4wIKXMDSoASAFQAw%3D%3D -->

# Multi-armed bandits

::: incremental
- Let us assume that we have a slot machine and we repeatedly can choose between $k$ options
-

:::