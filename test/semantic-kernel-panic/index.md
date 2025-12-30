---
title: 'Kernel Panic: Frustrations with Semantic Kernel'
date: 2024-09-21
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  # - "meta"
  - blogumentation
  - experiment
  # - "listicle"
  - opinion
  # ai/ml
  - generative AI
  # - "prompts"
  - LLMs
series: []
layout: wide
toc: true
math: false
draft: false
---

A few weeks ago at work, I wanted to ensure that the prompt template we used with Semantic Kernel transformed into the OpenAI API spec `messages` array that I expected.
Little did I know that this simple objective would take me a few days, several experimental notebooks, a thorough tour through the Semantic Kernel's dev blog and (limited) documentation,
and a review of pretty much the entire python package source code, unit tests, and example library.

Code from these experiments is available
[here](https://github.com/ahgraber/AIMLbling-about/tree/main/experiments/sk-rant).

## What is Semantic Kernel?

### Overview

> Semantic Kernel is a lightweight, open-source development kit that lets you easily build AI agents and integrate the latest AI models into your C#, Python, or Java codebase.
> It serves as an efficient middleware that enables rapid delivery of enterprise-grade solutions. [^semantic-kernel]

Competitors to Semantic Kernel include other Microsoft packages [Prompt Flow](https://microsoft.github.io/promptflow/index.html) and [AutoGen](https://microsoft.github.io/autogen/),
as well as more well-known packages in the Python + LLM ecosystem including [LangChain](https://langchain.com), [LlamaIndex](https://www.llamaindex.ai/), and [Haystack](https://haystack.deepset.ai/).

### Functionality

Semantic Kernel does a few nice things. It is the only AI development framework for C# or Java that I'm aware of.
It was early to the "function-calling" space, and as such has very good support for agentic-style workflows
that use LLMs to plan the work, determine which tools to use, create the arguments to call the function, and then _actually invoke the call_.
As a Microsoft project, Semantic Kernel has first class support for Azure integrations.
Finally (and most importantly to this article), Semantic Kernel allows the "chat" interaction method to be defined by templates,
which can make it easier for developers to dynamically build out interaction contexts with template variables. [^semantic-kernel]

The template functionality allows developers to manage, manipulate, and/or man-in-the-middle the input the user provides.
This can be extremely useful by providing a way to inject application context into the prompt text.
RAG (Retrieval Augmented Generation) is a common pattern in which information relevant to the user's request is added to the prompt context before the LLM generates a response.
This helps ensure the response is grounded and reduces the likelihood of hallucination.

{{% details title="RAG" closed="true" %}}

{{< figure
src="images/RAG.png"
alt="retrieval augmented generation"
link="https://docs.llamaindex.ai/en/stable/"
caption="Retrieval Augmented Generation. Credit: LlamaIndex" >}}
{{% /details %}}

## Kernel Panic

Before I get too much further, let me lay out my problem statement so that if a PM at Microsoft reads this post they at least have an easy copypasta for a work item:

> As a developer, I want an easy way to render the arguments that would be sent to the AI service _without actually having to make the call to the AI service_.
> When iterating on a template, this will allow me to see the exact effects of the changes I am making without spending money on the actual service call.

In other words, there should be an easy way (single function or method call) that provides insight as to whether the left and right cells below are truly equivalent.

<table>
<tr>
  <th>Semantic Kernel</th>
  <th>OpenAI</th>
</tr>
<tr>
  <td style="width:50%">

```py
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
kernel = Kernel()
service_id = "chat"
chat_service = OpenAIChatCompletion(service_id=service_id)
kernel.add_service(chat_service)

template = """
{{$system_message}}
{{$chat_history}}
{{$user_request}}
"""

system_message = "You are a helpful assistant."
chat_function = kernel.add_function(
    prompt=template,
    function_name="chat",
    plugin_name="chat",
    ...
)
chat_history = ChatHistory()
chat_history.add_user_message("Hi, who are you?")
chat_history.add_assistant_message("I am a helpful AI assistant.")
answer = await kernel.invoke(
    chat_function,
    chat_history=chat_history,
    user_input="""Why is the default program called "hello world"?""",
)
print(answer)
```

</td>
  <td style="width:50%">

```sh
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
      "role": "system",
      "content": "You are a helpful AI assistant."
      },
      {
      "role": "user",
      "content": "Hi, who are you?"
      },
      {
      "role": "assistant",
      "content": "I am a helpful AI assistant."
      },
      {
      "role": "user",
      "content": "Why is the default program called "hello world"?"
      }
    ],
    ...
  }'
```

</td>
</tr>
</table>

I am not the first person to suggest this; there is an [unresolved Github discussion about this very topic](https://github.com/microsoft/semantic-kernel/discussions/1239).
The discussion ultimately suggests using Semantic Kernel's OpenTelemetry support (not yet ready for Python) or Filters; both of these seem to require fully invoking the service call.

## Investigation and Manual Resolution

> [!NOTE]
> Caveats:
>
> 1. Semantic Kernel is still undergoing rapid development; this article is based on my experience developing with version 1.8.2 in early September, 2024.
> 2. I'm approaching this as a Python-focused data scientist/ML engineer, not as an application developer. I'd be unsurprised to find out that I missed something in my investigation that would have let me solve this faster.
> 3. Per (2) it's also possible I'm a shit dev... but I've also never seen inheritance spread so wide and deep such that it makes it nearly impossible to determine what parent class is providing the various methods.

I ended up doing a lot of extra work trying to figure out what Semantic Kernel is doing to go from a template to the API call.
The below image is from a call graph I made during the investigation.

{{< figure
src="images/trace-tree.png"
alt="traces"
caption="Simplified call graph of `kernel.invoke`" >}}

The most useful thing was actually reviewing the unit tests! Github issues and discussions did not provide much help, and the documentation was either too high level or pointed to dead pages.

Semantic Kernel renders templates to text with XML tags that represent the message roles, then converts the XML to Semantic Kernel's `ChatHistory` object that has direct conversion to the `messages` format used by most AI service APIs.
Since the internal representation of `ChatHistory` is XML, it is easy to go back-and-forth between the rendered template text and the `ChatHistory` object.
Note that the conversation's chat history can be inserted into the template; this is distinct from the internal chat history object that is used as the vehicle to convert from the XML rendered prompt to API-compatible `messages`.

### Template to API Call

Given the following setup:

```py
template = """
{{$system_message}}
{{$chat_history}}
{{$user_request}}
"""

system_message = "You are a helpful assistant."
chat_history = ChatHistory()
chat_history.add_user_message("Hi, who are you?")
chat_history.add_assistant_message("I am a helpful AI assistant.")

user_request = 'Why is the default program called "hello world"?'
```

The template is rendered to XML text, then converted to an internal `ChatHistory` object.
That internal `ChatHistory` is then easily converted to an API-compatible `messages` array.

<table>
<tr>
  <th>Semantic Kernel snippet</th>
  <th>Output</th>
</tr>
<tr>
  <td style="width:50%">

```py
# render template by injecting variable text
rendered = await chat_function.prompt_template.render(
    kernel,
    arguments= KernelArguments(req_settings,
                               system_message=system_message,
                               chat_history=chat_history,
                               user_request=user_request)
)

```

</td>
  <td style="width:50%">

```txt
You are a helpful AI assistant.
<chat_history>
<message role="user"><text>Hi, who are you?</text></message>
<message role="assistant"><text>I am a helpful AI assistant.</text></message>
</chat_history>
Why is the default program called "hello world"?
```

</td>
</tr>
<tr>
  <td>

```py
# convert rendered prompt to internal ChatHistory
history = ChatHistory.from_rendered_prompt(rendered)
# convert ChatHistory to messages array
messages = [message.to_dict() for message in history.messages]
```

</td>
  <td>

```json
[
  {
    "role": "system",
    "content": "You are a helpful AI assistant."
  },
  {
    "role": "user",
    "content": "Hi, who are you?"
  },
  {
    "role": "assistant",
    "content": "I am a helpful AI assistant."
  },
  {
    "role": "user",
    "content": "Why is the default program called \"hello world\"?"
  }
]
```

</td>
</tr>
</table>

### Gotchas and Potential Footguns

Semantic Kernel has internal logic that helps define user roles in circumstances when they may not be defined.
I remember seeing some mention about the first message defaulting to the `system` role, but I've been unable to rediscover that reference.
Thus, I've found the below logic experimentally during my investigation.

#### Inferred System Message

The system message can be specifically set by:

| Method                                            | Example                                  |
| ------------------------------------------------- | ---------------------------------------- |
| Adding it to the ChatHistory during instantiation | `ChatHistory(system_message="...")`      |
| Adding it to the ChatHistory after instantiation  | `chat_history.add_system_message("...")` |
| Labelling text in the template with the XML tag   | `<message role="system">...</message>`   |

#### Presence of ChatHistory differentiates System and User roles

In addition to the above methods, the system message is also inferred to be any text prior to the chat history, **_if_** the chat history is present **_and_** there is no previously-defined system message.
This holds true even if the chat history is empty.

<table>
<tr>
  <th>Template</th>
  <th>Output</th>
</tr>
<tr>
  <td style="width:50%">

```py
template = """
This would be a system message.
{{$chat_history}}
This would be a user message.
"""
```

</td>
  <td style="width:50%">

```json
[
  {
    "role": "system",
    "content": "This would be a system message."
  },
  // chat history here, if it exists
  {
    "role": "user",
    "content": "This would be a user message."
  }
]
```

</td>
</tr>
<tr>
  <td>

```py
template = """
This would be a system message.
This would be a user message.
"""
```

</td>
  <td>

```json
[
  // No chat history - single user message
  {
    "role": "user",
    "content": "This would be a system message.\nThis would be a user message."
  }
]
```

</td>
</tr>
</table>

However, if no chat history is provided, then the text is assumed to be a single user message.

## Conclusion

After exploring the codebase and running these experiments, I am of two minds with respect to Semantic Kernel.

On the one hand, I am impressed by and mostly approve of the design decisions.
I have a much better understanding of the capabilities Semantic Kernel has - even beyond the template-to-call introspection upon which this article focuses.
I also understand why the Semantic Kernel developers arrived at the (undefined) ruleset governing how roles are anticipated as the template gets transformed into messages.

However, I am still very frustrated with the user experience.
I don't understand how this functionality does not exist as a developer-facing convenience function - it seems like a critical oversight to be unable to introspect how the template is transformed to an API call.

I fully recognize that (a) keeping documentation in-sync with an evolving codebase is a huge challenge normally, and (b) supporting multiple languages means that someone is always ahead/behind.
That said, the documentation experience is _terrible_.
It is spread across the overview and API reference at [learn.microsoft.com](https://learn.microsoft.com/en-us/semantic-kernel/overview/),
the [Semantic Kernel devblog](https://devblogs.microsoft.com/semantic-kernel/), and [sample notebooks on github](https://github.com/microsoft/semantic-kernel/tree/main/python/samples).
In some cases, cross-linked references point to dead pages, and sometimes information is present in the dev blog that is not found elsewhere.

To quote [John Green](https://www.wnycstudios.org/podcasts/anthropocene-reviewed), "I rate Semantic Kernel 2 out of 5 stars."

## References

[^semantic-kernel]: [Introduction to Semantic Kernel | Microsoft Learn](https://learn.microsoft.com/en-us/semantic-kernel/overview/)
