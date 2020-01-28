# Language Modelling of Dialogue Act Sequences

This project evaluates the prediction performance of different models on Dialogue Act (DA) prediction given sequences of DAs. The models include baseline models (majority class, random class, weighted random class, bigram and trigram) and two LSTM models. The first LSTM model takes in the input features (speaker, DA, level, utterance length) and is called the 'old' model. The old model has a weighted and unweighted version. The second LSTM model has the same input features as the old model but with added text embedding. This is the 'new' model.

## Getting Started

To get this repository working, several python libraries need to be installed.

### Dependencies

numpy (1.16.5)
pandas (0.25.1)
pytorch (1.3.1)
json (2.0.9)
seaborn (0.9.0)
matplotlib (3.1.1)
scipy (1.3.1)
sklearn (0.21.3)
nltk (3.4.5)

##

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
