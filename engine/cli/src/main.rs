#![forbid(unsafe_code)]

#[cfg(all(feature = "handcrafted", feature = "pure-only"))]
compile_error!("handcrafted and pure-only are mutually exclusive");
#[cfg(not(any(feature = "handcrafted", feature = "pure-only")))]
compile_error!("select exactly one build class: handcrafted or pure-only");

use std::process::ExitCode;
mod arena_player;

// The filesystem workflows depend on the exact syscall surface reviewed in
// `open_shogi_core::secure_file`. `unix` alone also includes targets such as QNX/QURT for which
// that boundary has not been implemented, so keep the allow-list explicit and fail closed
// everywhere else.
macro_rules! secure_unix_items {
    ($($item:item)*) => {
        $(
            #[cfg(all(feature = "handcrafted", any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "watchos",
                target_os = "visionos",
                target_os = "linux",
                target_os = "android"
            )))]
            $item
        )*
    };
}

secure_unix_items! {
    mod arena;
    mod analysis;
    mod args;
    mod bench;
    mod checksum;
    mod dataset;
    mod model;
    mod opening;
    mod play;
    mod tools;

    use std::env;

    use open_shogi_core::EngineIdentity;

    fn main() -> ExitCode {
        let arguments = env::args().skip(1).collect::<Vec<_>>();
        match run(&arguments) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::from(2)
            }
        }
    }

    fn run(arguments: &[String]) -> Result<(), String> {
        let Some(command) = arguments.first().map(String::as_str) else {
            print_help();
            return Ok(());
        };
        match command {
            "--version" | "-V" if arguments.len() == 1 => {
                let identity = EngineIdentity::current();
                println!("{} {}", identity.name, identity.version);
                Ok(())
            }
            "usi" if arguments.len() == 1 => {
                open_shogi_usi::run_stdio().map_err(|error| format!("USI I/O failed: {error}"))
            }
            "perft" => tools::run_perft(&arguments[1..]),
            "random-games" => tools::run_random_games(&arguments[1..]),
            "validate-csa" => tools::run_validate_csa(&arguments[1..]),
            "export-csa-jsonl" => dataset::run(&arguments[1..]),
            "play" => play::run(&arguments[1..]),
            "arena" => arena::run(&arguments[1..]),
            "arena-player" => arena_player::run(&arguments[1..]),
            "analysis" => analysis::run(&arguments[1..]),
            "opening-book" => opening::run(&arguments[1..]),
            "model" => model::run(&arguments[1..]),
            "bench" => bench::run(&arguments[1..]),
            "help" | "--help" | "-h" if arguments.len() == 1 => {
                print_help();
                Ok(())
            }
            _ => Err("unknown command; run `open-shogi-cli help`".to_owned()),
        }
    }

    fn print_help() {
        println!("OpenShogiAI command-line tools");
        println!("  open-shogi-cli --version");
        println!("  open-shogi-cli usi");
        println!("  open-shogi-cli perft --depth N [--sfen SFEN] [--divide]");
        println!("  open-shogi-cli random-games [--games N] [--max-plies N] [--seed N]");
        println!("  open-shogi-cli validate-csa FILE");
        println!("  open-shogi-cli export-csa-jsonl --input-dir DIR --output FILE --max-games N");
        println!("  open-shogi-cli play --human black|white [search limits] [--output FILE]");
        println!(
            "  open-shogi-cli arena --games N --player-a TYPE --player-b TYPE [--git-commit SHA] [options]"
        );
        println!("  open-shogi-cli analysis  # newline-delimited analysis protocol on stdin/stdout");
        println!("  open-shogi-cli opening-book verify --book FILE");
        println!("  open-shogi-cli model inspect --model FILE");
        println!("  open-shogi-cli model infer --model FILE (--sfen SFEN | --input FILE)");
        println!(
            "  open-shogi-cli model infer-handcrafted --profile handcrafted-baseline|handcrafted-experimental (--sfen SFEN | --input FILE)"
        );
        println!("  open-shogi-cli bench [--depth N | --nodes N]");
    }

    #[cfg(test)]
    mod tests {
        use super::run;

        #[test]
        fn unknown_command_is_an_error() {
            assert!(run(&["unknown".to_owned()]).is_err());
        }
    }
}

#[cfg(all(
    feature = "handcrafted",
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "linux",
        target_os = "android"
    ))
))]
fn main() -> ExitCode {
    eprintln!(
        "error: open-shogi-cli filesystem workflows require a supported secure directory-descriptor boundary"
    );
    ExitCode::from(2)
}

#[cfg(feature = "pure-only")]
mod pure;

#[cfg(feature = "pure-only")]
fn main() -> ExitCode {
    match pure::run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
