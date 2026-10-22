using Parakeet.Numerics;

namespace Parakeet.Tests;

/// <summary>The fork/join worker pool behind <see cref="Parallelism"/>.</summary>
public sealed class WorkerPoolTests
{
    [Fact]
    public async Task WorkerPoolRunsEveryItemUnderStress()
    {
        // Jobs separated by gaps shorter and longer than the workers' spin window exercise the spin, sleep and wake
        // paths; a second caller thread exercises the concurrent-use fallback.
        using var par = new Parallelism(4);
        var task = Task.Run(() => Hammer(par, 7));
        Hammer(par, 3);
        await task.WaitAsync(TimeSpan.FromMinutes(2));

        static void Hammer(Parallelism par, int seed)
        {
            var rnd = new Random(seed);
            for (int job = 0; job < 1500; job++)
            {
                int count = rnd.Next(1, 40);
                var hits = new int[count];
                par.For(count, i => Interlocked.Increment(ref hits[i]));
                Assert.All(hits, h => Assert.Equal(1, h));
                int gap = rnd.Next(0, 4);
                if (gap == 1) Thread.SpinWait(rnd.Next(10, 2000));
                else if (gap == 2) Thread.Sleep(rnd.Next(0, 2));
            }
        }
    }
}
