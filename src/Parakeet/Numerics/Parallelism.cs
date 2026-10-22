namespace Parakeet.Numerics;

/// <summary>
/// Degree-of-parallelism control shared by every kernel. <c>1</c> runs everything inline on the calling thread
/// (the single-core configuration); larger values run work items on a dedicated <see cref="WorkerPool"/>.
/// </summary>
public sealed class Parallelism : IDisposable
{
    public static Parallelism Single { get; } = new(1);

    private readonly WorkerPool? _pool;

    public Parallelism(int maxThreads)
    {
        if (maxThreads <= 0) maxThreads = Environment.ProcessorCount;
        MaxThreads = maxThreads;
        Options = new ParallelOptions { MaxDegreeOfParallelism = maxThreads };
        if (maxThreads > 1) _pool = new WorkerPool(maxThreads - 1);
    }

    public int MaxThreads { get; }

    internal ParallelOptions Options { get; }

    /// <summary>Runs <paramref name="body"/> for every index in [0, <paramref name="count"/>).</summary>
    public void For(int count, Action<int> body)
    {
        if (count <= 0) return;
        if (_pool == null || count == 1)
        {
            for (int i = 0; i < count; i++) body(i);
            return;
        }
        if (!_pool.TryRun(count, body)) Parallel.For(0, count, Options, body);
    }

    /// <summary>Splits [0, <paramref name="total"/>) into at most <see cref="MaxThreads"/> contiguous ranges
    /// of at least <paramref name="minChunk"/> items.</summary>
    public void ForRange(int total, int minChunk, Action<int, int> body)
    {
        if (total <= 0) return;
        int chunks = Math.Max(1, Math.Min(MaxThreads, total / Math.Max(1, minChunk)));
        if (chunks == 1)
        {
            body(0, total);
            return;
        }
        int per = (total + chunks - 1) / chunks;
        For(chunks, c =>
        {
            int s = c * per;
            int e = Math.Min(total, s + per);
            if (s < e) body(s, e);
        });
    }

    public void Dispose() => _pool?.Dispose();
}

/// <summary>
/// A fork/join pool of dedicated worker threads for the many short parallel regions of a forward pass (hundreds of
/// GEMMs of a few milliseconds each). Workers spin briefly between jobs before blocking, so back-to-back regions start
/// within microseconds, and items are claimed one at a time with an atomic counter, so a late-waking thread never
/// leaves the others waiting at the join (as range partitioning over a handful of items can). Re-entrant or concurrent
/// use falls back to the caller (see <see cref="TryRun"/>).
/// </summary>
internal sealed class WorkerPool : IDisposable
{
    private static readonly long SpinTicks = System.Diagnostics.Stopwatch.Frequency / 10000;   // spin ≈ 100 µs before blocking

    private readonly Thread[] _threads;
    private readonly object _sleep = new();
    private readonly object _gate = new();
    private Action<int>? _body;
    private int _count;
    private int _next;
    private int _pending;          // participants (workers) still inside the current job
    private long _generation;
    private volatile bool _disposed;
    private int _sleepers;
    private Exception? _error;

    [ThreadStatic] private static bool t_inPool;

    public WorkerPool(int workers)
    {
        _threads = new Thread[workers];
        for (int i = 0; i < workers; i++)
        {
            _threads[i] = new Thread(Worker) { IsBackground = true, Name = $"Parakeet worker {i}" };
            _threads[i].Start();
        }
    }

    /// <summary>Runs the job on the pool plus the calling thread; returns false (without running anything) when called
    /// from inside a pool job or while another thread's job is running.</summary>
    public bool TryRun(int count, Action<int> body)
    {
        if (t_inPool || !Monitor.TryEnter(_gate)) return false;
        try
        {
            _body = body;
            _count = count;
            _error = null;
            Volatile.Write(ref _next, 0);
            Volatile.Write(ref _pending, _threads.Length);
            Interlocked.Increment(ref _generation);
            // Sleeping workers re-check the generation under the lock, so a pulse can never be lost (a semaphore
            // permit could be taken by a worker that already finished this job, leaving another one asleep).
            if (Volatile.Read(ref _sleepers) > 0)
                lock (_sleep) Monitor.PulseAll(_sleep);

            t_inPool = true;
            try
            {
                Work(body, count);
            }
            finally
            {
                t_inPool = false;
            }
            var spin = new SpinWait();
            while (Volatile.Read(ref _pending) > 0) spin.SpinOnce(-1);
            _body = null;
            if (_error != null) throw new AggregateException(_error);
            return true;
        }
        finally
        {
            Monitor.Exit(_gate);
        }
    }

    private void Work(Action<int> body, int count)
    {
        int i;
        while ((i = Interlocked.Increment(ref _next) - 1) < count)
        {
            try
            {
                body(i);
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref _error, e, null);
            }
        }
    }

    private void Worker()
    {
        t_inPool = true;
        long seen = 0;
        while (!_disposed)
        {
            long gen = Volatile.Read(ref _generation);
            if (gen == seen)
            {
                // Spin briefly (back-to-back GEMMs arrive within microseconds), then block until the next job.
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                while ((gen = Volatile.Read(ref _generation)) == seen && !_disposed &&
                       System.Diagnostics.Stopwatch.GetTimestamp() - start < SpinTicks)
                    Thread.SpinWait(1);
                if (gen == seen && !_disposed)
                {
                    lock (_sleep)
                    {
                        Interlocked.Increment(ref _sleepers);
                        while ((gen = Volatile.Read(ref _generation)) == seen && !_disposed) Monitor.Wait(_sleep);
                        Interlocked.Decrement(ref _sleepers);
                    }
                }
                if (_disposed) return;
            }
            seen = gen;
            var body = _body;
            if (body != null) Work(body, _count);
            Interlocked.Decrement(ref _pending);
        }
    }

    public void Dispose()
    {
        _disposed = true;
        lock (_sleep) Monitor.PulseAll(_sleep);
    }
}
