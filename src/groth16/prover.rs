use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use ff::{Field, PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::info;
use rand_core::RngCore;

use super::{ParameterSource, Proof, VerifyingKey};
use crate::bls::Engine;
use crate::domain::{EvaluationDomain, Scalar};
use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::multicore::{Worker, RAYON_THREAD_POOL, THREAD_POOL};
use crate::multiexp::{multiexp, DensityTracker, FullDensity};
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable, BELLMAN_VERSION,
};

#[cfg(feature = "gpu")]
use crate::gpu::PriorityLock;

fn eval<E: Engine>(
    lc: &LinearCombination<E>,
    mut input_density: Option<&mut DensityTracker>,
    mut aux_density: Option<&mut DensityTracker>,
    input_assignment: &[E::Fr],
    aux_assignment: &[E::Fr],
) -> E::Fr {
    let mut acc = E::Fr::zero();

    for (&index, &coeff) in lc.0.iter() {
        let mut tmp;

        match index {
            Variable(Index::Input(i)) => {
                tmp = input_assignment[i];
                if let Some(ref mut v) = input_density {
                    v.inc(i);
                }
            }
            Variable(Index::Aux(i)) => {
                tmp = aux_assignment[i];
                if let Some(ref mut v) = aux_density {
                    v.inc(i);
                }
            }
        }

        if coeff == E::Fr::one() {
            acc.add_assign(&tmp);
        } else {
            tmp.mul_assign(&coeff);
            acc.add_assign(&tmp);
        }
    }

    acc
}

struct ProvingAssignment<E: Engine> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar<E>>,
    b: Vec<Scalar<E>>,
    c: Vec<Scalar<E>>,

    // Assignments of variables
    input_assignment: Vec<E::Fr>,
    aux_assignment: Vec<E::Fr>,
}

impl<E: Engine> ProvingAssignment<E> {
    fn from_circuit<C: Circuit<E>>(circuit: C) -> Result<Self, SynthesisError> {
        let mut pa = Self::new();
        pa.alloc_input(|| "", || Ok(E::Fr::one()))?;

        circuit.synthesize(&mut pa)?;

        for i in 0..pa.input_assignment.len() {
            pa.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
        }
        Ok(pa)
    }
}

impl<E: Engine> fmt::Debug for ProvingAssignment<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ProvingAssignment")
            .field("a_aux_density", &self.a_aux_density)
            .field("b_input_density", &self.b_input_density)
            .field("b_aux_density", &self.b_aux_density)
            .field(
                "a",
                &self
                    .a
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "b",
                &self
                    .b
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "c",
                &self
                    .c
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field("input_assignment", &self.input_assignment)
            .field("aux_assignment", &self.aux_assignment)
            .finish()
    }
}

impl<E: Engine> PartialEq for ProvingAssignment<E> {
    fn eq(&self, other: &ProvingAssignment<E>) -> bool {
        self.a_aux_density == other.a_aux_density
            && self.b_input_density == other.b_input_density
            && self.b_aux_density == other.b_aux_density
            && self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.input_assignment == other.input_assignment
            && self.aux_assignment == other.aux_assignment
    }
}

impl<E: Engine> ConstraintSystem<E> for ProvingAssignment<E> {
    type Root = Self;

    fn new() -> Self {
        Self {
            a_aux_density: DensityTracker::new(),
            b_input_density: DensityTracker::new(),
            b_aux_density: DensityTracker::new(),
            a: vec![],
            b: vec![],
            c: vec![],
            input_assignment: vec![],
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        self.a.push(Scalar(eval(
            &a,
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(&mut self.a_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.b.push(Scalar(eval(
            &b,
            Some(&mut self.b_input_density),
            Some(&mut self.b_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.c.push(Scalar(eval(
            &c,
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            &self.input_assignment,
            &self.aux_assignment,
        )));
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: Self) {
        self.a_aux_density.extend(other.a_aux_density, false);
        self.b_input_density.extend(other.b_input_density, true);
        self.b_aux_density.extend(other.b_aux_density, false);

        self.a.extend(other.a);
        self.b.extend(other.b);
        self.c.extend(other.c);

        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(other.aux_assignment);
    }
}

pub fn create_random_proof_batch_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
    R: RngCore,
{
    let r_s = (0..circuits.len()).map(|_| E::Fr::random(rng)).collect();
    let s_s = (0..circuits.len()).map(|_| E::Fr::random(rng)).collect();

    create_proof_batch_priority::<E, C, P>(circuits, params, r_s, s_s, priority)
}

pub fn create_proof_batch_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
{
    let proofs = RAYON_THREAD_POOL.install(|| {
        create_proof_batch_priority_inner::<E, C, P>(circuits, params, r_s, s_s, priority)
    })?;

    Ok(proofs)
}

fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    mut circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
{
    info!("Bellperson {} is being used!", BELLMAN_VERSION);
    if circuits.is_empty() {
        return Ok(Vec::new());
    }

    // Start prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let worker = Worker::new();
    let num_circuits = circuits.len();

    // Synthesize first circuit
    info!("synthesize first circuit");
    let prover = ProvingAssignment::from_circuit(circuits.pop().expect("at least one circuit"))?;

    let input_len = prover.input_assignment.len();
    let vk = params.get_vk(input_len)?.clone();
    let n = prover.a.len();

    info!("log_d");
    let mut log_d = 0;
    while (1 << log_d) < n {
        log_d += 1;
    }

    #[cfg(feature = "gpu")]
    let prio_lock = if priority {
        log::trace!("acquiring priority lock");
        Some(PriorityLock::lock())
    } else {
        None
    };

    let params = &params;
    let mut proofs = Vec::with_capacity(num_circuits);
    let worker = &worker;
    let proofs_ref = &mut proofs;
    let r_s = &r_s;
    let s_s = &s_s;

    // Now do 1 FFT while synthesizing the next circuit on a second thread.
    THREAD_POOL.scoped(|s| {
        // We use a rendevouz channel to ensure only one fft and one synthesis
        // are performed in parallel.
        let (sender, receiver) = std::sync::mpsc::sync_channel(0);

        s.execute(move || {
            while let Some(circuit) = circuits.pop() {
                info!("synthesize {} start", circuits.len());
                let p = ProvingAssignment::from_circuit(circuit);
                info!("synthesize {} done", circuits.len());
                let is_err = p.is_err();
                // TODO: verify that the circuit sizes all match with the first circuit.
                sender.send(p).unwrap();
                if is_err {
                    break;
                }
            }
        });

        s.recurse(move |s| {
            // First Round
            match round::<E, P>(
                worker, log_d, priority, &vk, &r_s[0], &s_s[0], prover, params, s,
            ) {
                Ok(proof) => {
                    proofs_ref.push(Ok(proof));
                    let mut i = 1;
                    while let Ok(prover) = receiver.recv() {
                        match prover {
                            Ok(prover) => match round::<E, P>(
                                worker, log_d, priority, &vk, &r_s[i], &s_s[i], prover, params, s,
                            ) {
                                Ok(proof) => {
                                    proofs_ref.push(Ok(proof));
                                }
                                Err(err) => {
                                    proofs_ref.push(Err(err));
                                    break;
                                }
                            },
                            Err(err) => {
                                proofs_ref.push(Err(err));
                                break;
                            }
                        }
                        i += 1;
                    }
                }
                Err(err) => {
                    proofs_ref.push(Err(err));
                    return;
                }
            }
        });
    });

    let proofs = proofs
        .into_iter()
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    assert_eq!(proofs.len(), num_circuits);

    #[cfg(feature = "gpu")]
    {
        log::trace!("dropping priority lock");
        drop(prio_lock);
    }

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

macro_rules! load {
    ($params:expr, $scope:expr, $f:tt) => {{
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        $scope.execute(move || {
            sender.send($f($params)).unwrap();
        });

        Waiter { receiver }
    }};
}

fn round<'a, E, P>(
    worker: &Worker,
    log_d: usize,
    priority: bool,
    vk: &VerifyingKey<E>,
    r: &E::Fr,
    s: &E::Fr,
    mut prover: ProvingAssignment<E>,
    params: &'a P,
    scope: &yastl::Scope<'a>,
) -> Result<Proof<E>, SynthesisError>
where
    E: Engine,
    P: ParameterSource<E> + 'a,
{
    info!("round start");
    let input_len = prover.input_assignment.len();
    let a_aux_density_total = prover.a_aux_density.get_total_density();
    let b_input_density_total = prover.b_input_density.get_total_density();
    let b_aux_density_total = prover.b_aux_density.get_total_density();
    let n = prover.a.len();

    // load params for the next multiexp (non blocking)
    let h_waiter = load!(params, scope, (|params: &P| params.get_h(n)));
    let a = execute_fft(&worker, &mut prover, log_d, priority)?;
    let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(log_d, priority));

    info!("multiexp h");
    let params_h = h_waiter.wait()?;
    // load params for the next multiexp (non blocking)
    let l_waiter = load!(params, scope, (|params: &P| params.get_l(n)));
    let h = multiexp(&worker, params_h, FullDensity, a, &mut multiexp_kern)?;

    let ia: Vec<_> = std::mem::replace(&mut prover.input_assignment, Vec::new())
        .into_iter()
        .map(|s| s.into_repr())
        .collect();
    let ia = Arc::new(ia);

    let aux: Vec<_> = std::mem::replace(&mut prover.aux_assignment, Vec::new())
        .into_iter()
        .map(|s| s.into_repr())
        .collect();
    let aux = Arc::new(aux);

    info!("multiexp l");
    let params_l = l_waiter.wait()?;
    // load params for the next multiexp (non blocking)
    let a_source_waiter = load!(
        params,
        scope,
        (|params: &P| params.get_a(input_len, a_aux_density_total))
    );

    let l = multiexp(
        &worker,
        params_l,
        FullDensity,
        aux.clone(),
        &mut multiexp_kern,
    )?;

    info!("multiexp a_inputs");
    let (a_inputs_source, a_aux_source) = a_source_waiter.wait()?;
    // load params for the next multiexp (non blocking)
    let b_g1_source_waiter = load!(
        params,
        scope,
        (|params: &P| params.get_b_g1(b_input_density_total, b_aux_density_total))
    );

    let a_inputs = multiexp(
        &worker,
        a_inputs_source,
        FullDensity,
        ia.clone(),
        &mut multiexp_kern,
    )?;

    info!("multiexp a_aux");
    let a_aux = multiexp(
        &worker,
        a_aux_source,
        Arc::new(prover.a_aux_density),
        aux.clone(),
        &mut multiexp_kern,
    )?;

    let b_input_density = Arc::new(prover.b_input_density);
    let b_aux_density = Arc::new(prover.b_aux_density);
    let (b_g1_inputs_source, b_g1_aux_source) = b_g1_source_waiter.wait()?;

    // load params for the next multiexp (non blocking)
    let b_g2_source_waiter = load!(
        params,
        scope,
        (|params: &P| params.get_b_g2(b_input_density_total, b_aux_density_total))
    );

    info!("multiexp b_g1_inputs");
    let b_g1_inputs = multiexp(
        &worker,
        b_g1_inputs_source,
        b_input_density.clone(),
        ia.clone(),
        &mut multiexp_kern,
    )?;

    info!("multiexp b_g1_aux");
    let b_g1_aux = multiexp(
        &worker,
        b_g1_aux_source,
        b_aux_density.clone(),
        aux.clone(),
        &mut multiexp_kern,
    )?;

    let (b_g2_inputs_source, b_g2_aux_source) = b_g2_source_waiter.wait()?;

    info!("multiexp b_g2_inputs");
    let b_g2_inputs = multiexp(
        &worker,
        b_g2_inputs_source,
        b_input_density,
        ia,
        &mut multiexp_kern,
    )?;

    info!("multiexp b_g2_aux");
    let b_g2_aux = multiexp(
        &worker,
        b_g2_aux_source,
        b_aux_density,
        aux,
        &mut multiexp_kern,
    )?;

    info!("proofs prep");
    if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
        // If this element is zero, someone is trying to perform a
        // subversion-CRS attack.
        return Err(SynthesisError::UnexpectedIdentity);
    }

    let mut g_a = vk.delta_g1.mul(*r);
    g_a.add_assign_mixed(&vk.alpha_g1);
    let mut g_b = vk.delta_g2.mul(*s);
    g_b.add_assign_mixed(&vk.beta_g2);
    let mut g_c;
    {
        let mut rs = *r;
        rs.mul_assign(&s);

        g_c = vk.delta_g1.mul(rs);
        g_c.add_assign(&vk.alpha_g1.mul(*s));
        g_c.add_assign(&vk.beta_g1.mul(*r));
    }

    let mut a_answer = a_inputs;
    a_answer.add_assign(&a_aux);
    g_a.add_assign(&a_answer);
    a_answer.mul_assign(*s);
    g_c.add_assign(&a_answer);

    let mut b1_answer = b_g1_inputs;
    b1_answer.add_assign(&b_g1_aux);
    let mut b2_answer = b_g2_inputs;
    b2_answer.add_assign(&b_g2_aux);

    g_b.add_assign(&b2_answer);
    b1_answer.mul_assign(*r);
    g_c.add_assign(&b1_answer);
    g_c.add_assign(&h);
    g_c.add_assign(&l);

    let proof = Proof {
        a: g_a.into_affine(),
        b: g_b.into_affine(),
        c: g_c.into_affine(),
    };

    info!("round end");
    Ok(proof)
}

fn execute_fft<E: Engine>(
    worker: &Worker,
    prover: &mut ProvingAssignment<E>,
    log_d: usize,
    priority: bool,
) -> Result<Arc<Vec<<<E as ScalarEngine>::Fr as PrimeField>::Repr>>, SynthesisError> {
    let mut fft_kern = Some(LockedFFTKernel::<E>::new(log_d, priority));

    let mut a = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
    let mut b = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;

    info!("a: ifft");
    a.ifft(&worker, &mut fft_kern)?;
    a.coset_fft(&worker, &mut fft_kern)?;

    info!("b: ifft");
    b.ifft(&worker, &mut fft_kern)?;
    b.coset_fft(&worker, &mut fft_kern)?;

    a.mul_assign(&worker, &b);
    drop(b);

    info!("c: ifft");
    let mut c = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

    c.ifft(&worker, &mut fft_kern)?;
    c.coset_fft(&worker, &mut fft_kern)?;

    a.sub_assign(&worker, &c);
    drop(c);

    info!("coset");
    a.divide_by_z_on_coset(&worker);
    a.icoset_fft(&worker, &mut fft_kern)?;

    let a = a.into_coeffs();
    let a_len = a.len() - 1;

    Ok(Arc::new(
        a.into_iter()
            .take(a_len)
            .map(|s| s.0.into_repr())
            .collect::<Vec<_>>(),
    ))
}

struct Waiter<L> {
    receiver: std::sync::mpsc::Receiver<Result<L, SynthesisError>>,
}
impl<L> Waiter<L> {
    pub fn wait(self) -> Result<L, SynthesisError> {
        self.receiver.recv().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bls::{Bls12, Fr};
    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_proving_assignment_extend() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut full_assignment = ProvingAssignment::<Bls12>::new();
                full_assignment
                    .alloc_input(|| "one", || Ok(Fr::one()))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(Fr::one())).unwrap();
                        partial_assignments.push(p)
                    }

                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    // TODO: LinearCombination
                }

                let mut combined = ProvingAssignment::new();
                combined.alloc_input(|| "one", || Ok(Fr::one())).unwrap();

                for assignment in partial_assignments.into_iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}
